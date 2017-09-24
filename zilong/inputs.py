from utils import *


def read_batch(filename_queue):
    reader = tf.TFRecordReader()
    _, raw = reader.read(filename_queue)

    batch = tf.train.shuffle_batch(
        [raw],
        batch_size=FLAGS.batch_size,
        num_threads=12,
        capacity=200000,
        min_after_dequeue=20000,
        allow_smaller_final_batch=True
    )
    features = tf.parse_example(
        batch,
        features={
            'label': tf.FixedLenFeature([1], tf.int64),
            'deep': tf.VarLenFeature(tf.int64),
            'wide': tf.VarLenFeature(tf.int64),
            'iid': tf.FixedLenFeature([1], tf.int64),
        }
    )
    return features
