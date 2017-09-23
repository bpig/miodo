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
    train_dir = "data/test/"

    # data_file = [ int(_[-5:]) for _ in os.listdir(train_dir) if _.startswith("part")]
    # num = sorted(data_file)
    # print num, len(num)
    # for i in range(1000):
    #     assert i == num[i], i

    data_file = [train_dir + _ for _ in os.listdir(train_dir) if _.startswith("part")]
    # valid_dir = "data/validate/"
    # data_file += [valid_dir + _ for _ in os.listdir(valid_dir) if _.startswith("part")]
    # data_file = data_file[:2]
    print len(data_file)
    data = _read_by_queue(data_file, 2048)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        fout = open("pred.fea", "w")
        fl = open("pred.lb", "w")        
        keys = []
        c = 0
        try:
            while not coord.should_stop():
                label, fid, iid = sess.run([data['label'], data['fid'], data['iid']])
                label = label.reshape(-1)
                iid = iid.values.reshape(-1)
                row = fid.indices[:, 0]
                col = fid.values
                idx = 0
                for i in range(label.shape[0]):
                    print >> fout, label[i],
                    while idx < row.shape[0] and row[idx]  == i:
                        print >> fout, str(col[idx]) + ":1",
                        idx += 1
                    print >> fout
                    print >> fl, iid[i]
                c += len(label)
                print time.ctime(), len(label), c
        except tf.errors.OutOfRangeError:
            print "finsh read data"
        finally:
            coord.request_stop()
            coord.join(threads)


def train():
    X = load_npz("train.npz")[:10]
    print X.shape
    y = np.fromfile("train.label", dtype=np.int64)[:10]
    print y.shape

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.95, random_state=1314)

    gbct = GradientBoostingClassifier(max_depth=2, n_estimators=30, warm_start=True)
    gbct.fit(X_train, y_train)
    model = gbct.get_params()
    pickle.dump(model, "gbct.ml")



def split():
    import xgboost as xgb
    dm = xgb.DMatrix("train.xgb")
    row = dm.num_row()
    ct = int(row * 0.95)
    print ct, row
    row = range(row)
    random.shuffle(row)
    
    train = dm.slice(row[:ct])
    test = dm.slice(row[ct:])

    train.save_binary("tr.xgb")
    test.save_binary("te.xgb")    


if __name__ == "__main__":
    trans_data()
