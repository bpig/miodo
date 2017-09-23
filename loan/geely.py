# coding:utf-8

from common import *
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz

feature_map = {
    'label': tf.FixedLenFeature([1], tf.int64),
    'fid': tf.VarLenFeature(tf.int64),
    'iid': tf.VarLenFeature(tf.string),
}


def _read_by_queue(data_file, batch_size=2048, num_epochs=1):
    filename_queue = tf.train.string_input_producer(
        data_file, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)

    batch = tf.train.shuffle_batch(
        [value],
        batch_size=batch_size,
        num_threads=12,
        capacity=500000,
        min_after_dequeue=100000,
        allow_smaller_final_batch=True
    )

    return tf.parse_example(batch, features=feature_map)


def load_data():
    data_file = []
    data = _read_by_queue(data_file)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        fids = []
        labels = []
        try:
            while not coord.should_stop():
                label, fid = sess.run([data['label'], data['fid']])
                fids += [fid]
                labels += [label]
        except tf.errors.OutOfRangeError:
            print "finsh read data"
        finally:
            coord.request_stop()
            coord.join(threads)

        concat = tf.sparse_concat(1, fids)
        data = sess.run(concat)
        label = np.concatenate(labels, 1)
    print data.dense_shape
    print data.indices
    print data.values

    print label.shape


def train():
    # coo_matrix((data, (i, j)), [shape=(M, N)])

    # coo_matrix()
    # gbct = GradientBoostingClassifier(warm_start=True)
    # gbct.fit()
    pass


if __name__ == "__main__":
    load_data()
    pass
