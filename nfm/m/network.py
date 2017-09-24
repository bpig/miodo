from utils import *

import math


def inference_deep_wide(deep_feature_index, deep_feature_id, wide_feature_index, wide_feature_id, instance_id, dims, keep_prob=1):
  deep_features = tf.sparse_merge(deep_feature_index, deep_feature_id, FLAGS.num_deep_features, name=None, already_sorted=False)
  wide_features = tf.sparse_merge(wide_feature_index, wide_feature_id, FLAGS.num_wide_features, name=None, already_sorted=False)

  deep_logits, deep_predict = nn_layers(deep_features, None, FLAGS.num_deep_features, dims, keep_prob)
  wide_logits, wide_predict = wide_layers(wide_features, None, FLAGS.num_wide_features)

  with tf.variable_scope('tot_sigmoid'):
    #logits = deep_logits + wide_logits
    logits = wide_logits
    predict = tf.nn.sigmoid(logits)

    logits = tf.clip_by_value(logits, -4, 4)
    predict = tf.clip_by_value(predict, 0.02, 0.98)
  return logits, predict, instance_id


def nn_layers(ids, values, num_features, dims, keep_prob=1):
  hidden1_units = dims[0]
  hidden2_units = dims[1]
  hidden3_units = dims[2]
  with tf.variable_scope("embedding"):
    weights_shape = [num_features, hidden1_units]
    biases_shape = [hidden1_units,]
    hidden1 = sparse_embedding(ids, values, weights_shape, biases_shape)
    hidden1 = tf.nn.relu(hidden1)

  # Hidden 2
  with tf.variable_scope('hidden2'):
    hidden2 = tf.nn.relu(full_connect_layer(hidden1, hidden1_units, hidden2_units))
    hidden2 = tf.nn.dropout(hidden2, keep_prob)

  # Hidden 3
  with tf.variable_scope('hidden3'):
    hidden3 = tf.nn.relu(full_connect_layer(hidden2, hidden2_units, hidden3_units))
    hidden3 = tf.nn.dropout(hidden3, keep_prob)

  # Sigmoid output
  with tf.variable_scope('sigmoid'):
    logits = full_connect_layer(hidden3, hidden3_units, 1)
  return logits, tf.nn.sigmoid(logits)


def wide_layers(ids, values, num_features):
  with tf.variable_scope("wide_hidden1"):
    weights_shape = [num_features, 1]
    biases_shape = [1,]
    hidden1 = sparse_embedding(ids, values, weights_shape, biases_shape, True)
  return  hidden1, tf.nn.sigmoid(hidden1)


def sparse_embedding(ids, values, weights_shape, biases_shape, zero_init=False):
  num_shards = 1
  init=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(weights_shape[0])))

  if zero_init:
    init = tf.zeros_initializer

  weights = tf.get_variable("weights",
                            weights_shape,
                            partitioner=tf.fixed_size_partitioner(num_shards, axis=0),
                            initializer=init)

  biases = tf.get_variable("biases",
                           biases_shape,
                           initializer=tf.zeros_initializer)

  result = tf.nn.embedding_lookup_sparse(weights, ids, values, partition_strategy="div", combiner="sum") + biases
  return result


def full_connect_layer(inputs, layer1_units, layer2_units):
  weights, biases = weights_and_biases(layer1_units, layer2_units)
  return tf.matmul(inputs, weights) + biases


def weights_and_biases(layer1_units, layer2_units, zero_init=False):
  init = tf.truncated_normal(
    [layer1_units, layer2_units],
    stddev=1.0/math.sqrt(float(layer1_units)),
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

  sum_loss = tf.reduce_sum(loss)
  mean_loss = tf.reduce_mean(loss)

  sum_loss = tf.Print(sum_loss, [sum_loss], "sum_loss: ", first_n=100, summarize=200)
  mean_loss = tf.Print(mean_loss, [mean_loss], "mean_loss: ", first_n=100, summarize=200)
  return sum_loss, mean_loss
