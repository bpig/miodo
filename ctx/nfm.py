# coding:utf-8

from common import *


class NFM(NET):
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        'fid': tf.VarLenFeature(tf.int64),
        'fval': tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    def inference(self, fea, drop=0.5):
        fea = fea['fid']

        self._initialize_weights()
        weights = self.weights

        segment_ids = fea.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        ids = fea.values
        ids, idx = tf.unique(ids)

        embed = tf.nn.embedding_lookup(weights['emb_weight'], ids)

        square_last_embed = tf.square(
            tf.sparse_segment_mean(embed, idx, segment_ids))
        mean_last_embed = tf.sparse_segment_mean(
            tf.square(embed), idx, segment_ids)

        fm = 0.5 * tf.subtract(square_last_embed, mean_last_embed)
        fm = tf.nn.dropout(fm, keep_prob)

        for i in range(0, len(self.layer_dim)):
            fm = tf.add(tf.matmul(fm, weights['l%d' % i]), weights['b%d' % i])
            fm = tf.nn.relu(fm)
            fm = tf.nn.dropout(fm, keep_prob)

        fm = tf.matmul(fm, weights['pred']) + weights['bias']
        emb_bias = tf.nn.embedding_lookup_sparse(weights['emb_bias'], fea, None, combiner="mean")
        return fm + emb_bias

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # tf.layers.batch_normalization()
        bn_train = tf.layers.batch_normalization(x, momentum=0.9, center=True, scale=True, training=True)
        # bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
        #                           is_training=False, reuse=True, trainable=True, scope=scope_bn)
        # z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return tf.no_op()

    def _initialize_weights(self):
        all_weights = dict()
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.sparse_dim)))
        all_weights['emb_weight'] = tf.get_variable(
            shape=[self.sparse_dim, self.hidden_factor], initializer=init, name='feature_embeddings')
        all_weights['emb_bias'] = tf.get_variable(
            shape=[self.sparse_dim, 1], initializer=tf.zeros_initializer, name='feature_bias')

        assert self.layer_dim
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.hidden_factor)))
        all_weights['l0'] = tf.get_variable(
            name="l0", initializer=init, shape=[self.hidden_factor, self.layer_dim[0]], dtype=tf.float32)
        all_weights['b0'] = tf.get_variable(
            name="b0", initializer=tf.zeros_initializer, shape=[self.layer_dim[0]], dtype=tf.float32)
        for i in range(1, len(self.layer_dim)):
            init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.layer_dim[i])))
            all_weights['l%d' % i] = tf.get_variable(
                name="l%d" % i, initializer=init, shape=self.layer_dim[i - 1:i + 1], dtype=tf.float32)
            all_weights['b%d' % i] = tf.get_variable(
                name="b%d" % i, initializer=tf.zeros_initializer, shape=[self.layer_dim[i]], dtype=tf.float32)
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.layer_dim[-1])))
        all_weights['pred'] = tf.get_variable(
            name="pred", initializer=init, shape=[self.layer_dim[-1], 1], dtype=tf.float32)
        all_weights['bias'] = tf.Variable(0.0, name='bias')
        self.weights = all_weights


if __name__ == "__main__":
    pass
