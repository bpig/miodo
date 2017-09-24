from utils import *

import math


def inference_deep_wide(fea, drop=0.5):
    deep_fea = fea['deep']
    wide_fea = fea['wide']
    deep_logits = deep(deep_fea, 2064805, drop)
    wide_logits = wide(wide_fea, 66040165)

    with tf.variable_scope('merge'):
        logits = deep_logits + wide_logits

    return logits


def deep(ids, dense_dim, dims, drop=0.5):
    dims = [64, 64, 64]
    with tf.variable_scope("embedding"):
        weights_shape = [dense_dim, dims[0]]
        biases_shape = dims[:1]
        layer = sparse_embedding(ids, weights_shape, biases_shape)
        layer = tf.nn.relu(layer)

    with tf.variable_scope('deep'):
        for i in range(1, len(dims)):
            init = tf.truncated_normal_initializer(
                stddev=1.0 / math.sqrt(float(dims[i - 1])), seed=100007)
            layer = tf.layers.dense(layer, dims[i], activation=tf.nn.relu, kernel_initializer=init)
            layer = tf.nn.dropout(layer, drop)

    with tf.variable_scope('logits'):
        init = tf.truncated_normal_initializer(
            stddev=1.0 / math.sqrt(float(dims[- 1])), seed=100007)
        logits = tf.layers.dense(layer, 1, kernel_initializer=init)
    return logits


def wide(ids, wide_dim):
    with tf.device("/cpu:0"), tf.variable_scope("wide"):
        weights_shape = [wide_dim, 1]
        biases_shape = [1, ]
        layer = sparse_embedding(ids, weights_shape, biases_shape)
    return layer


def sparse_embedding(ids, weights_shape, biases_shape):
    init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(weights_shape[0])))

    weights = tf.get_variable("weights", weights_shape, initializer=init)
    biases = tf.get_variable("biases", biases_shape, initializer=tf.zeros_initializer)

    return tf.nn.embedding_lookup_sparse(weights, ids, None, combiner="sum") + biases


def loss_op(logits, labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels), logits=logits)
    return tf.reduce_mean(loss)
