from args import *

import math


def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)


def inference_deep_wide(fea, dims, keep_prob=1):
    if FLAGS.format == "old":
        deep_features, wide_features, iid = fea['deep_feature_id'], fea['wide_feature_id'], fea['instance_id']
    else:
        deep_features, wide_features, iid = fea['deep'], fea['wide'], fea['iid'],

    deep_logits = nn_layers(deep_features, None, FLAGS.deep_dim, dims, keep_prob)
    wide_logits = wide_layers(wide_features, None, FLAGS.wide_dim)

    with tf.variable_scope('output'):
        logits = deep_logits + wide_logits
        predict = tf.nn.sigmoid(logits)
    return logits, predict, iid


def nn_layers(ids, values, num_features, dims, keep_prob=1):
    hidden1_units = dims[0]
    hidden2_units = dims[1]
    hidden3_units = dims[2]
    with tf.variable_scope("embed"):
        weights_shape = [num_features, hidden1_units]
        biases_shape = [hidden1_units, ]
        hidden1 = sparse_embedding(ids, values, weights_shape, biases_shape)
        hidden1 = leaky_relu(hidden1)

    with tf.variable_scope('hidden2'):
        hidden2 = leaky_relu(full_connect_layer(hidden1, hidden1_units, hidden2_units))
        hidden2 = tf.nn.dropout(hidden2, keep_prob)

    with tf.variable_scope('hidden3'):
        hidden3 = leaky_relu(full_connect_layer(hidden2, hidden2_units, hidden3_units))
        hidden3 = tf.nn.dropout(hidden3, keep_prob)

    with tf.variable_scope('sigmoid'):
        logits = full_connect_layer(hidden3, hidden3_units, 1)
    return logits


def wide_layers(ids, values, num_features):
    with tf.variable_scope("wide"):
        weights_shape = [num_features, 1]
        biases_shape = [1, ]
        return sparse_embedding(ids, values, weights_shape, biases_shape, True)


def sparse_embedding(ids, values, weights_shape, biases_shape, zero_init=False):
    init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(weights_shape[0])))

    weights = tf.get_variable("weights",
                              weights_shape,
                              initializer=init)

    biases = tf.get_variable("biases",
                             biases_shape,
                             initializer=tf.zeros_initializer)

    return tf.nn.embedding_lookup_sparse(weights, ids, values, combiner="sum") + biases


def full_connect_layer(inputs, layer1_units, layer2_units):
    weights, biases = weights_and_biases(layer1_units, layer2_units)
    return tf.matmul(inputs, weights) + biases


def weights_and_biases(layer1_units, layer2_units, zero_init=False):
    init = tf.truncated_normal(
        [layer1_units, layer2_units],
        stddev=1.0 / math.sqrt(float(layer1_units)),
        dtype=tf.float32,
        seed=100007
    )
    if zero_init:
        init = tf.zeros([layer1_units, layer2_units])
    weights = tf.get_variable(name='weights', initializer=init)
    biases = tf.get_variable(name='biases', shape=[layer2_units], initializer=tf.zeros_initializer, dtype=tf.float32)
    return weights, biases


def log_loss(logits, labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels), logits=logits)

    return tf.reduce_mean(loss)


def calc_deep_wide_metrics(sess, batch):
    label = tf.placeholder(tf.float32)

    deep_feature_index = tf.sparse_placeholder(tf.int64)
    deep_feature_id = tf.sparse_placeholder(tf.int64)

    wide_feature_index = tf.sparse_placeholder(tf.int64)
    wide_feature_id = tf.sparse_placeholder(tf.int64)

    instance_id = tf.placeholder(tf.int64)

    tf.get_variable_scope().reuse_variables()
    logits, predict, _ = inference_deep_wide(deep_feature_id, wide_feature_id,
                                             instance_id, [])

    sum_loss, _ = log_loss(logits, label)
    _, auc_update_op = tf.contrib.metrics.streaming_auc(predict, label, num_thresholds=1000)

    feature_num = len(batch) * FLAGS.batch_size

    sess.run(tf.local_variables_initializer())

    feature_sum_loss = 0
    batch_auc = 0
    for sample in batch:
        cur_sum_loss, batch_auc, pred_result = sess.run(
            [sum_loss, auc_update_op, predict],
            feed_dict={label: sample['label'],
                       deep_feature_index: sample['deep_feature_index'],
                       deep_feature_id: sample['deep_feature_id'],
                       wide_feature_index: sample['wide_feature_index'],
                       wide_feature_id: sample['wide_feature_id'],
                       instance_id: sample['instance_id']
                       }
        )
        feature_sum_loss += cur_sum_loss
    return feature_sum_loss, feature_sum_loss / feature_num, batch_auc
