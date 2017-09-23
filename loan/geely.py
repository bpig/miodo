# coding:utf-8

from common import *
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

feature_map = {
    'label': tf.FixedLenFeature(1, tf.int64),
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


def trans_data():
    # top_dir = "data/"
    train_dir = "data/train/"
    data_file = [train_dir + _ for _ in os.listdir(train_dir) if _.startswith("part")]
    valid_dir = "data/validate/"
    data_file += [valid_dir + _ for _ in os.listdir(valid_dir) if _.startswith("part")]
    # data_file = data_file[:2]
    print len(data_file)
    data = _read_by_queue(data_file, 2048)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        labels = []
        row = []
        col = []
        try:
            while not coord.should_stop():
                label, fid = sess.run([data['label'], data['fid']])
                label = label.reshape(-1)
                row += [fid.indices[:, 0] + len(labels)]
                labels += list(label)                
                col += [fid.values]
                # if len(labels) % 5000 == 0:
                print time.ctime(), len(labels)
        except tf.errors.OutOfRangeError:
            print "finsh read data"
        finally:
            coord.request_stop()
            coord.join(threads)

    
    row = np.concatenate(row, 0)
    col = np.concatenate(col, 0)
    data = np.ones(len(row))
    coo = coo_matrix((data, (row, col)))
    print coo.shape
    save_npz("train", coo)

    labels = np.asarray(labels)    
    labels.tofile(open("train.label", "w"))


def train():
    # coo_matrix((data, (i, j)), [shape=(M, N)])

    # coo_matrix()
    gbct = GradientBoostingClassifier(warm_start=True)
    gbct.fit()
    pass


if __name__ == "__main__":
    trans_data()
    # print np.ones(3)
    pass
