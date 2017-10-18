from args import *
from common import *


def list_dir(top_dir, key):
    prefix = top_dir + "/" + key
    ans = [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    random.shuffle(ans)
    return ans


def read_pred():
    top_dir = "opt1_0.1/"
    pred_dir = list_dir(top_dir, "test/")
    print "pred dir len", len(pred_dir)
    fq = tf.train.string_input_producer(pred_dir, num_epochs=1)
    return read_batch(fq)


def read_data():
    top_dir = "opt1_0.1/"

    train_dir = list_dir(top_dir, "train/")
    valid_dir = list_dir(top_dir, "test/")

    print "train dir len", len(train_dir)
    print "valid dir len", len(valid_dir)

    fq = tf.train.string_input_producer(train_dir, num_epochs=100)
    fea = read_batch(fq)

    fq = tf.train.string_input_producer(valid_dir, num_epochs=None)
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
    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        '1f': tf.VarLenFeature(tf.int64),
        '2f': tf.VarLenFeature(tf.int64),
        '3f': tf.VarLenFeature(tf.int64),
        '4f': tf.VarLenFeature(tf.int64),
        '5f': tf.VarLenFeature(tf.int64),
        '6f': tf.VarLenFeature(tf.int64),
        '7f': tf.VarLenFeature(tf.int64),
        '8f': tf.VarLenFeature(tf.int64),
        '9f': tf.VarLenFeature(tf.int64),
        '10f': tf.VarLenFeature(tf.int64),
        '11f': tf.VarLenFeature(tf.int64),
        '12f': tf.VarLenFeature(tf.int64),
        'iid': tf.VarLenFeature(tf.string),
    }
    features = tf.parse_example(
        batch,
        features=feature_map
    )
    return features
