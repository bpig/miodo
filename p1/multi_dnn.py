# coding:utf-8
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
    #    "dense": 49,
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
        "dense_val": tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    def gen_embed(self, fea, sparse_dim, name, embed_dim=16, commbiner="mean"):
        # glorot = tf.uniform_unit_scaling_initializer
        glorot = tf.truncated_normal_initializer(stddev=0.02)
        weights = tf.get_variable("w_" + name, [sparse_dim, embed_dim],
                                  initializer=glorot)
        biases = tf.get_variable("b" + name, [embed_dim], initializer=tf.zeros_initializer)

        segment_ids = fea.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        ids = fea.values
        ids, idx = tf.unique(ids)

        embed = tf.nn.embedding_lookup(weights, ids)
        square_last_embed = tf.square(
            tf.sparse_segment_mean(embed, idx, segment_ids))
        mean_last_embed = tf.sparse_segment_mean(
            tf.square(embed), idx, segment_ids)
        return 0.5 * tf.subtract(square_last_embed, mean_last_embed) + biases

        # return tf.nn.embedding_lookup_sparse(weights, fea, None, combiner=commbiner) + biases

    def inference(self, fea):
        glorot = tf.truncated_normal_initializer(stddev=0.02)
        dense_dim = 50
        dense = tf.sparse_merge(fea['dense_id'], fea['dense_val'], dense_dim, name="sp_merge")
        dense = tf.sparse_tensor_to_dense(dense, name="sp_to_dense")
        dense = tf.to_float(dense, "float_dense")

        embeds = []
        total_dim = 0
        with tf.variable_scope("embed"):
            embed_dim = 32
            for key in sparse_table.keys():
                embed = self.gen_embed(fea[key + "_id"], sparse_table[key] + 1, key, embed_dim)
                total_dim += embed_dim
                embeds += [embed]
            embeds += [dense]
            embed = tf.concat(embeds, axis=1)
            total_dim += dense_dim

        with tf.variable_scope("deep"):
            w = tf.get_variable(name="w0", shape=(total_dim, self.layer_dim[0]), initializer=glorot)
            b = tf.get_variable(name="b0", shape=(self.layer_dim[0]), initializer=tf.zeros_initializer)
            pre_layer = tf.matmul(embed, w) + b
            pre_layer = tf.nn.relu(pre_layer)

            def reg(x):
                return 0.1 * tf.nn.l2_loss(x)

            for i in range(1, len(self.layer_dim)):
                layer = tf.layers.dense(pre_layer, self.layer_dim[i], name="layer%d" % i,
                                        activation=tf.nn.relu, kernel_initializer=glorot)
                pre_layer = layer

        with tf.variable_scope("concat"):
            logits = tf.layers.dense(pre_layer, 1, name="logists",
                                     kernel_initializer=glorot)
            logits = logits + dense

        return logits


if __name__ == "__main__":
    pass
