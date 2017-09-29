# coding:utf-8

from common import *


class DNNFC(NET):
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        'fid': tf.VarLenFeature(tf.int64),
        'iid': tf.VarLenFeature(tf.string),
    }

    def inference(self, fea, drop=0.4):
        self.ema_factor = 0.992
        self.step = 10
        
        fea = fea['fid']

        with tf.variable_scope("deep"):
            pre_layer = fea
            for i in range(len(self.layer_dim)):
                pre_dim = self.layer_dim[i - 1] if i > 0 else self.sparse_dim
                pre_dim = float(pre_dim)
                init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(pre_dim))
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
            # logits = tf.clip_by_value(logits, -5, 5)

        return logits


if __name__ == "__main__":
    pass
