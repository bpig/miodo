# coding:utf-8

from common import *


class FTRL(NET):
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        'fid': tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    def inference(self, fea, drop=0.4):
        self.step = 10
        fea = fea['fid']

        with tf.variable_scope("wide"):
            init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.sparse_dim)))
            weights = tf.get_variable("weights", [self.sparse_dim, 1],
                                      initializer=init)
            biases = tf.get_variable("biases", [1], initializer=tf.zeros_initializer)
            embed = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="sum") + biases

        return embed


if __name__ == "__main__":
    pass
