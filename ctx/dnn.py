# coding:utf-8
# 2017/9/19 下午1:34
# 286287737@qq.com
from common import *


class DNN(NET):
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        'fid': tf.VarLenFeature(tf.int64),
#        'fval': tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    def inference(self, fea):
        glorot = tf.uniform_unit_scaling_initializer
        fea = fea['fid']
        # with tf.variable_scope("wide"):
        #     weights = tf.get_variable(
        #         "weights", [sparse_dim, 1], initializer=glorot)
        #     biases = tf.get_variable(
        #         "biases", [1], initializer=tf.zeros_initializer)
        #     wide = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="sum") + biases

        with tf.variable_scope("embed"):
            weights = tf.get_variable("weights", [self.sparse_dim, self.layer_dim[0]],
                                      initializer=glorot)
            biases = tf.get_variable("biases", [self.layer_dim[0]], initializer=tf.zeros_initializer)

            embed = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="mean") + biases

        with tf.variable_scope("deep"):
            pre_layer = embed
            for i in range(1, len(self.layer_dim)):
                layer = tf.layers.dense(pre_layer, self.layer_dim[i], name="layer%d" % i,
                                        activation=tf.nn.elu, kernel_initializer=glorot)
                pre_layer = layer

        with tf.variable_scope("concat"):
            logits = tf.layers.dense(pre_layer, 1, name="logists",
                                     kernel_initializer=glorot)

        return logits


if __name__ == "__main__":
    pass
