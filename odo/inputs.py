from args import *
from common import *


def list_dir(top_dir, begin, end):
    ans = []
    for d in range(begin, end + 1):
        prefix = "%s/date=%2d/" % (top_dir, d)
        ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    random.shuffle(ans)
    return ans


def read_data():
    top_dir = FLAGS.top_dir

    train_dir = list_dir(top_dir, FLAGS.train_begin, FLAGS.train_end)
    valid_dir = list_dir(top_dir, FLAGS.valid_begin, FLAGS.valid_end)

    print "train dir len", len(train_dir)
    print "valid dir len", len(valid_dir)

    fq = tf.train.string_input_producer(train_dir)
    fea = read_batch(fq)

    fq = tf.train.string_input_producer(valid_dir, num_epochs=1)
    valid_fea = read_batch(fq)
    return fea, valid_fea


def read_all_batch(filename_queue):
    features = read_batch(filename_queue)
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
    return parse_example(batch)


def parse_example(batch):
    features = tf.parse_example(
        batch,
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
