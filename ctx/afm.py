# coding:utf-8
# 2017/9/18 下午8:01
# 286287737@qq.com

from common import *


class AFM(NET):
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        'dense': tf.VarLenFeature(tf.int64),
        # 'fval': tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    def inference(self, fea, drop=0.5):
        fea = fea['dense']
        self.att_dim = self.hidden_factor
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
        fm = tf.layers.dropout(fm, drop)
        # attention model

        att_state = tf.add(tf.matmul(fm, weights['att_W']), weights['att_bias'])
        att_state = tf.nn.relu(att_state)
        att_state = tf.matmul(att_state, weights['att_h'])

        att_sum = tf.reduce_sum(att_state, 1, keep_dims=True)
        att_state = tf.div(att_state, att_sum)

        fm = tf.multiply(fm, att_state)

        for i in range(0, len(self.layer_dim)):
            fm = tf.add(tf.matmul(fm, weights['l%d' % i]), weights['b%d' % i])
            fm = tf.nn.relu(fm)
            fm = tf.layers.dropout(fm, drop)

        fm = tf.matmul(fm, weights['pred']) + weights['bias']
        emb_bias = tf.nn.embedding_lookup_sparse(weights['emb_bias'], fea, None, combiner="mean")
        return fm + emb_bias

    def _initialize_weights(self):
        all_weights = dict()
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.sparse_dim)))
        all_weights['emb_weight'] = tf.get_variable(
            shape=[self.sparse_dim, self.hidden_factor], initializer=init, name='feature_embeddings')

        all_weights['emb_bias'] = tf.get_variable(
            shape=[self.sparse_dim, 1], initializer=tf.zeros_initializer, name='feature_bias')

        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.att_dim)))
        all_weights['att_h'] = tf.get_variable(
            shape=[self.att_dim, 1], initializer=init, name='att_h')
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.hidden_factor)))
        all_weights['att_W'] = tf.get_variable(
            shape=[self.hidden_factor, self.att_dim], initializer=init, name='att_W')
        all_weights['att_bias'] = tf.get_variable(
            shape=[self.att_dim], initializer=tf.zeros_initializer, name='att_bias')

        assert self.layer_dim
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.hidden_factor)))
        all_weights['l0'] = tf.get_variable(
            name="l0", initializer=init, shape=[self.hidden_factor, self.layer_dim[0]], dtype=tf.float32)
        all_weights['b0'] = tf.get_variable(
            name="b0", initializer=tf.zeros_initializer, shape=[self.layer_dim[0]], dtype=tf.float32)

        for i in range(1, len(self.layer_dim)):
            init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.layer_dim[i - 1])))
            all_weights['l%d' % i] = tf.get_variable(
                name="l%d" % i, initializer=init, shape=self.layer_dim[i - 1:i + 1], dtype=tf.float32)
            all_weights['b%d' % i] = tf.get_variable(
                name="b%d" % i, initializer=tf.zeros_initializer, shape=[self.layer_dim[i]], dtype=tf.float32)

        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.layer_dim[- 1])))
        all_weights['pred'] = tf.get_variable(
            name="pred", initializer=init, shape=[self.layer_dim[-1], 1], dtype=tf.float32)

        all_weights['bias'] = tf.Variable(0.0, name='bias')
        self.weights = all_weights


if __name__ == "__main__":
    pass
