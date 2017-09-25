from args import *
import time


def read_data():
    ans = []
    for d in range(11, 28):
        prefix = "data/date=%2d/" % d
        ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    random.shuffle(ans)

    train_dir = ans

    print "train dir len", len(train_dir)

    ans = []
    for d in range(29, 31):
        prefix = "data/date=%2d/" % d
        ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    all_valid_files = ans

    print "valid dir len", len(all_valid_files)


def read_all_batch(filename_queue):
    features = read_batch(filename_queue, 0)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    data = []

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            while True:
                cur_features = sess.run(features)
                data.append(cur_features)
                if len(data) % 1000 == 0:
                    print time.ctime(), 'loading data, len: %d' % len(data)
                    logger.info('loading data, len: %d' % (len(data)))
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
    logger.info('load data finished, ct: %d' % len(data) * FLAGS.batch_size)
    return data


def read_batch(filename_queue, is_shuffle):
    serialized_example = read_and_decode(filename_queue)

    min_after_dequeue = 20000
    capacity = 200000
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
