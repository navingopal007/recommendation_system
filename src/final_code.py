import argparse
import numpy as np
from time import time
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
from abc import abstractmethod
import os

def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation

def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    interaction_table, offset = get_interaction_table(train_data, n_entity)
    model = KGNN_LS(args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        interaction_table.init.run()

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)

            # CTR evaluation
            train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)

            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

            # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')


# interaction_table is used for fetching user-item interaction label in LS regularization
# key: user_id * 10^offset + item_id
# value: y_{user_id, item_id}
def get_interaction_table(train_data, n_entity):
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.astype(np.int64)
    values = train_data[:, 2].astype(np.float32)

    interaction_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5)
    return interaction_table, offset


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

class KGNN_LS(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset):
        self._parse_args(args, adj_entity, adj_relation, interaction_table, offset)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation, interaction_table, offset):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        # LS regularization
        self.interaction_table = interaction_table
        self.offset = offset
        self.ls_weight = args.ls_weight

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGNN_LS.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGNN_LS.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGNN_LS.get_initializer(), name='relation_emb_matrix')

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # LS regularization
        self._build_label_smoothness_loss(entities, relations)

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    # feature propagation
    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = SumAggregator(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = SumAggregator(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    # LS regularization
    def _build_label_smoothness_loss(self, entities, relations):

        # calculate initial labels; calculate updating masks for label propagation
        entity_labels = []
        reset_masks = []  # True means the label of this item is reset to initial value during label propagation
        holdout_item_for_user = None

        for entities_per_iter in entities:
            # [batch_size, 1]
            users = tf.expand_dims(self.user_indices, 1)
            # [batch_size, n_neighbor^i]
            user_entity_concat = users * self.offset + entities_per_iter

            # the first one in entities is the items to be held out
            if holdout_item_for_user is None:
                holdout_item_for_user = user_entity_concat

            # [batch_size, n_neighbor^i]
            initial_label = self.interaction_table.lookup(user_entity_concat)
            holdout_mask = tf.cast(holdout_item_for_user - user_entity_concat, tf.bool)  # False if the item is held out
            reset_mask = tf.cast(initial_label - tf.constant(0.5), tf.bool)  # True if the entity is a labeled item
            reset_mask = tf.logical_and(reset_mask, holdout_mask)  # remove held-out items
            initial_label = tf.cast(holdout_mask, tf.float32) * initial_label + tf.cast(
                tf.logical_not(holdout_mask), tf.float32) * tf.constant(0.5)  # label initialization

            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        reset_masks = reset_masks[:-1]  # we do not need the reset_mask for the last iteration

        # label propagation
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        aggregator = LabelAggregator(self.batch_size, self.dim)
        for i in range(self.n_iter):
            entity_labels_next_iter = []
            for hop in range(self.n_iter - i):
                vector = aggregator(self_vectors=entity_labels[hop],
                                    neighbor_vectors=tf.reshape(
                                        entity_labels[hop + 1], [self.batch_size, -1, self.n_neighbor]),
                                    neighbor_relations=tf.reshape(
                                        relation_vectors[hop], [self.batch_size, -1, self.n_neighbor, self.dim]),
                                    user_embeddings=self.user_embeddings,
                                    masks=reset_masks[hop])
                entity_labels_next_iter.append(vector)
            entity_labels = entity_labels_next_iter

        self.predicted_labels = tf.squeeze(entity_labels[0], axis=-1)

    def _build_train(self):
        # base loss
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # L2 loss
        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        # LS loss
        self.ls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.predicted_labels))
        self.loss += self.ls_weight * self.ls_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)


LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class LabelAggregator(Aggregator):
    def __init__(self, batch_size, dim, name=None):
        super(LabelAggregator, self).__init__(batch_size, dim, 0., None, name)

    def _call(self, self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks):
        # [batch_size, 1, 1, dim]
        user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

        # [batch_size, -1, n_neighbor]
        user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
        user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1]
        neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_labels, axis=-1)
        output = tf.cast(masks, tf.float32) * self_labels + tf.cast(
            tf.logical_not(masks), tf.float32) * neighbors_aggregated

        return output
np.random.seed(555)

parser = argparse.ArgumentParser()


# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=1.0, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')

show_loss = False
show_time = False
show_topk = False

t = time()

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)

if show_time:
    print('time used: %d s' % (time() - t))
