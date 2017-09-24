from utils import *


def read_all_batch(filename_queue):
  features = read_batch(filename_queue, 0)
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  data = []
  line_count = 0
  print("FLAGS.num_threads %d", FLAGS.num_threads)
  config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
      while True:
        cur_features = sess.run(features)
        data.append(cur_features)
        line_count += len(cur_features)
        if len(data) % 1000 == 0:
          logger.info('loading data, len: %d' % (len(data)))
    except tf.errors.OutOfRangeError as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)
  logger.info('load data finished, len: %d, line: %d' % (len(data), len(data) * FLAGS.batch_size))
  return data


def read_batch(filename_queue, is_shuffle):
  print("FLAGS.num_threads %d", FLAGS.num_threads)
  serialized_example = read_and_decode(filename_queue)

  min_after_dequeue = 5000
  capacity = 100000
  batch_serialized_example = tf.train.batch(
    [serialized_example],
    batch_size=FLAGS.batch_size,
    num_threads=FLAGS.num_threads,
    capacity=capacity,
    allow_smaller_final_batch=True
  )

  if is_shuffle:
    batch_serialized_example = tf.train.shuffle_batch(
      [serialized_example],
      batch_size=FLAGS.batch_size,
      num_threads=FLAGS.num_threads,
      capacity=capacity,
      min_after_dequeue=min_after_dequeue,
      allow_smaller_final_batch=True
    )
  return parse_example(batch_serialized_example)


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  return serialized_example


def parse_example(batch_serialized_example):
  features = tf.parse_example(
    batch_serialized_example,
    features={
      'label': tf.FixedLenFeature([1], tf.int64),
      'deep_feature_index': tf.VarLenFeature(tf.int64),
      'deep_feature_id': tf.VarLenFeature(tf.int64),
      'wide_feature_index': tf.VarLenFeature(tf.int64),
      'wide_feature_id': tf.VarLenFeature(tf.int64),
      'instance_id': tf.FixedLenFeature([1], tf.int64),
    }
  )
  return features
