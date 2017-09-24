from network import *

def calc_deep_wide_metrics(sess, batch):
  label = tf.placeholder(tf.float32)

  deep_feature_index = tf.sparse_placeholder(tf.int64)
  deep_feature_id    = tf.sparse_placeholder(tf.int64)

  wide_feature_index = tf.sparse_placeholder(tf.int64)
  wide_feature_id    = tf.sparse_placeholder(tf.int64)

  instance_id = tf.placeholder(tf.int64)

  tf.get_variable_scope().reuse_variables()
  logits, predict, _ = inference_deep_wide(deep_feature_index, deep_feature_id, wide_feature_index, wide_feature_id, instance_id, layers)

  sum_loss, _ = log_loss(logits, label)
  _, auc_update_op = tf.contrib.metrics.streaming_auc(predict, label, num_thresholds=1000)

  feature_num = len(batch) * FLAGS.batch_size

  sess.run(tf.local_variables_initializer())  # streaming_auc need local variables
  logger.info('batch length: %d', len(batch))
  logger.info('feature number: %d', feature_num)

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
  return feature_sum_loss, feature_sum_loss/feature_num, batch_auc

