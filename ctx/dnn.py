# coding:utf-8

from common import *


def max_norm_regularizer(threshold=1.0, axes=1, name="max_norm", collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None

    return max_norm


class DNN(NET):
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        'fid': tf.VarLenFeature(tf.int64),
        #        'fval': tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    def inference(self, fea, drop=0.4):
        fea = fea['fid']

        batch_norm_layer = partial(tf.layers.batch_normalization,
                                   training=self.training, momentum=0.9)

        with tf.variable_scope("embed"):
            init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.sparse_dim)))
            # init = tf.zeros_initializer
            weights = tf.get_variable("weights", [self.sparse_dim, self.layer_dim[0]],
                                      initializer=init)
            biases = tf.get_variable("biases", [self.layer_dim[0]], initializer=tf.zeros_initializer)
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            embed = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="sum") + biases

        with tf.variable_scope("deep"):
            pre_layer = embed
            for i in range(1, len(self.layer_dim)):
                init = tf.truncated_normal_initializer(
                    stddev=1.0 / math.sqrt(float(self.layer_dim[i - 1])))
                # init = tf.uniform_unit_scaling_initializer(1.43)
                layer = tf.layers.dense(pre_layer, self.layer_dim[i], name="layer%d" % i,
                                        activation=self.leaky_relu,
                                        kernel_regularizer=max_norm_regularizer,
                                        kernel_initializer=init)
                layer = tf.layers.dropout(layer, drop)
                pre_layer = layer
                # tf.summary.histogram("weights", weights)
                # tf.summary.histogram("biases", biases)

        with tf.variable_scope("concat"):
            init = tf.truncated_normal_initializer(
                stddev=1.0 / math.sqrt(float(self.layer_dim[-1])))
            # init = tf.uniform_unit_scaling_initializer(1.)
            logits = tf.layers.dense(pre_layer, 1, name="logists",
                                     kernel_initializer=init)

        return logits


if __name__ == "__main__":
    pass
