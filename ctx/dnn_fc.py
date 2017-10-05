# coding:utf-8

from common import *


class DNNFC(NET):
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        # 'fid': tf.VarLenFeature(tf.int64),
        'wide': tf.VarLenFeature(tf.int64),
        'deep': tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    def load_ftrl_weight(self, filename):
        weights = [0]
        for l in open(filename):
            l = l.strip()
            if not l:
                continue
            items = l.split()
            assert len(items) == 3
            w = float(items[2])
            weights += [w]
        assert len(weights) == self.sparse_dim, len(weights)
        weights = np.asarray(weights, dtype=np.float32).reshape((-1, 1))
        return weights

    def inference(self, fea, drop=0.4):
        self.step = 10
        w_fea = fea['wide']
        fea = fea['deep']

        with tf.variable_scope("ftrl"):
            bias = self.cf.getfloat("net", "w_bias")
            weight_file = self.cf.get("net", "w_weight")
            # bias = -0.614403
            ftrl_weight = self.load_ftrl_weight(weight_file)
            weights = tf.Variable(ftrl_weight, name="ftrl_weight", trainable=False)
            ftrl = tf.nn.embedding_lookup_sparse(weights, w_fea, None, combiner="sum") + bias

        batch_norm_layer = partial(tf.layers.batch_normalization,
                                   training=self.training, momentum=0.9)

        with tf.variable_scope("embed"):
            init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.sparse_dim)))
            weights = tf.get_variable("weights", [self.sparse_dim, self.layer_dim[0]],
                                      initializer=init)
            biases = tf.get_variable("biases", [self.layer_dim[0]],
                                     initializer=tf.zeros_initializer()
            )
            embed = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="sum") + biases
            embed = self.leaky_relu(embed)

        with tf.variable_scope("deep"):
            pre_layer = embed
            for i in range(1, len(self.layer_dim)):
                init = tf.truncated_normal_initializer(
                    stddev=1.0 / math.sqrt(float(self.layer_dim[i - 1])))
                layer = tf.layers.dense(pre_layer, self.layer_dim[i], name="layer%d" % i,
                                        activation=self.leaky_relu,
                                        kernel_initializer=init)
                layer = tf.layers.dropout(layer, drop)
                pre_layer = layer

        with tf.variable_scope("concat"):
            init = tf.truncated_normal_initializer(
                stddev=1.0 / math.sqrt(float(self.layer_dim[-1])))
            logits = tf.layers.dense(pre_layer, 1, name="logists",
                                     kernel_initializer=init)
            logits += ftrl

        return logits


if __name__ == "__main__":
    pass
