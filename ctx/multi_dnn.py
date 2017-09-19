# coding:utf-8
# 2017/9/19 下午1:34
# 286287737@qq.com
from common import *

sparse_table = {
    "adid": 2163,
    "adtz": 62,
    "adcp": 354,
    "adc1": 16,
    "adc2": 43,
    "adap": 167,
    "adnm": 1494,
    "adaa": 49,
    "ad1a": 16,
    "ad2a": 37,
    "uspv": 136,
    "usct": 654,
    "usdi": 121,
    "usia": 72557,
    "usin": 316,
    "usua": 103077,
    "usuo": 251646,
    "usut": 374605,
    "usai": 158644,
    "usau": 153916,
    "usas": 166669,
    "usat": 277,
    "coip": 748827,
    "coav": 1097,
    "dense": 49,
}


class MultiDNN(NET):
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        "adid_id": tf.VarLenFeature(tf.int64),
        "adtz_id": tf.VarLenFeature(tf.int64),
        "adcp_id": tf.VarLenFeature(tf.int64),
        "adc1_id": tf.VarLenFeature(tf.int64),
        "adc2_id": tf.VarLenFeature(tf.int64),
        "adap_id": tf.VarLenFeature(tf.int64),
        "adnm_id": tf.VarLenFeature(tf.int64),
        "adaa_id": tf.VarLenFeature(tf.int64),
        "ad1a_id": tf.VarLenFeature(tf.int64),
        "ad2a_id": tf.VarLenFeature(tf.int64),
        "uspv_id": tf.VarLenFeature(tf.int64),
        "usct_id": tf.VarLenFeature(tf.int64),
        "usdi_id": tf.VarLenFeature(tf.int64),
        "usia_id": tf.VarLenFeature(tf.int64),
        "usin_id": tf.VarLenFeature(tf.int64),
        "usua_id": tf.VarLenFeature(tf.int64),
        "usuo_id": tf.VarLenFeature(tf.int64),
        "usut_id": tf.VarLenFeature(tf.int64),
        "usai_id": tf.VarLenFeature(tf.int64),
        "usau_id": tf.VarLenFeature(tf.int64),
        "usas_id": tf.VarLenFeature(tf.int64),
        "usat_id": tf.VarLenFeature(tf.int64),
        "coip_id": tf.VarLenFeature(tf.int64),
        "coav_id": tf.VarLenFeature(tf.int64),
        "dense_id": tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    def gen_embed(self, fea, sparse_dim, name):
        embed_dim = 8
        glorot = tf.uniform_unit_scaling_initializer
        weights = tf.get_variable("w_" + name, [sparse_dim, embed_dim],
                                  initializer=glorot)
        biases = tf.get_variable("b" + name, [embed_dim], initializer=tf.zeros_initializer)
        return tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="mean") + biases

    def inference(self, fea):
        glorot = tf.uniform_unit_scaling_initializer

        embeds = []
        with tf.variable_scope("embed"):
            for key in sparse_table.keys():
                embed = self.gen_embed(fea[key + "_id"], sparse_table[key], key)
                embeds += [embed]

        embed = tf.concat(embeds, axis=1)

        with tf.variable_scope("deep"):
            pre_layer = embed
            for i in range(len(self.layer_dim)):
                layer = tf.layers.dense(pre_layer, self.layer_dim[i], name="layer%d" % i,
                                        activation=tf.nn.elu, kernel_initializer=glorot)
                pre_layer = layer

        with tf.variable_scope("concat"):
            logits = tf.layers.dense(pre_layer, 1, name="logists",
                                     kernel_initializer=glorot)

        return logits


if __name__ == "__main__":
    pass
