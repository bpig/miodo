# coding:utf-8

from common import *


class WDE(NET):
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        # 'wide': tf.VarLenFeature(tf.int64),
        # 'deep': tf.VarLenFeature(tf.int64),
        'fid': tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    def inference(self, fea, drop=0.4):
        wide_fea = fea['fid']
        deep_fea = fea['fid']

        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.sparse_dim)))
        with tf.device("/cpu:0"), tf.variable_scope("wide"):
            weights = tf.get_variable(
                "weights", [self.sparse_dim, 1], initializer=init)
            biases = tf.get_variable(
                "biases", [1], initializer=tf.zeros_initializer)
            wide = tf.nn.embedding_lookup_sparse(weights, wide_fea, None, combiner="sum") + biases

        # wide_dim = 2650256
        wide_dim = self.sparse_dim
        with tf.variable_scope("embed"):
            init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(wide_dim)))
            weights = tf.get_variable("weights", [wide_dim, self.layer_dim[0]],
                                      initializer=init)
            biases = tf.get_variable("biases", [self.layer_dim[0]], initializer=tf.zeros_initializer)
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            embed = tf.nn.embedding_lookup_sparse(weights, deep_fea, None, combiner="sum") + biases

        with tf.variable_scope("deep"):
            pre_layer = embed
            for i in range(1, len(self.layer_dim)):
                init = tf.truncated_normal_initializer(
                    stddev=1.0 / math.sqrt(float(self.layer_dim[i - 1])))
                layer = tf.layers.dense(pre_layer, self.layer_dim[i], name="layer%d" % i,
                                        activation=tf.nn.relu, kernel_initializer=init)
                layer = tf.layers.dropout(layer, drop)
                pre_layer = layer

        with tf.variable_scope("output"):
            init = tf.truncated_normal_initializer(
                stddev=1.0 / math.sqrt(float(self.layer_dim[-1])))
            logits = tf.layers.dense(pre_layer, 1, name="logists",
                                     kernel_initializer=init)
            logits += wide

        return logits


if __name__ == "__main__":
    pass
