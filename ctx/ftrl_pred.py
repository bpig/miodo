# coding:utf-8
# 2017/9/30 下午1:15
# 286287737@qq.com
from common import *


class FtrlPred:
    @staticmethod
    def get_data_list():
        ans = []
        for d in range(29, 30 + 1):
            prefix = "%s/date=%2d/" % ("opt8_100/", d)
            print prefix
            ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
        random.shuffle(ans)
        return ans

    @staticmethod
    def read(feature_map):
        ans = FtrlPred.get_data_list()
        print len(ans)
        return FtrlPred._read_by_queue(ans, 1, 1024, feature_map)

    @staticmethod
    def _read_by_queue(data_file, num_epochs, batch_size, feature_map, name="train"):
        filename_queue = tf.train.string_input_producer(
            data_file, num_epochs=num_epochs, name=name)
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

    feature_map = {
        'label': tf.FixedLenFeature([1], tf.int64),
        'fid': tf.VarLenFeature(tf.int64),
        'iid': tf.FixedLenFeature(1, tf.int64),
    }

    @staticmethod
    def load_ftrl_weight(filename):
        weights = [-0.614403]  # * self.sparse_dim
        for l in open(filename):
            l = l.strip()
            if not l:
                continue
            items = l.split()
            assert items
            # idx = int(items[1])
            w = float(items[2])
            # weights[idx] = w
            weights += [w]

        weights = np.asarray(weights, dtype=np.float32).reshape((-1, 1))
        print weights.shape
        print weights.mean(), weights.std()
        return weights

    @staticmethod
    def inference(fea):
        fea = fea['fid']

        with tf.variable_scope("ftrl"):
            ftrl_weight = FtrlPred.load_ftrl_weight("fea2id_opt1_100")
            weights = tf.Variable(ftrl_weight, name="ftrl_weight", trainable=False)
            fea = tf.sparse_to_indicator(fea, 28677)
            fea = tf.to_float(fea)
            ftrl = tf.matmul(fea, weights) - 0.614403
        return tf.sigmoid(ftrl)

    @staticmethod
    def dump_pred(ans):
        label, prob, iid = [_.reshape(-1) for _ in ans]

        for v1, v2, v3 in zip(label, prob, iid):
            print v1, v2, v3

    @staticmethod
    def run():
        kv = FtrlPred.read(FtrlPred.feature_map)
        pred = FtrlPred.inference(kv)
        gpu_options = tf.GPUOptions(allow_growth=True)

        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            try:
                while not coord.should_stop():
                    ans = sess.run([kv['label'], pred, kv['iid']])
                    FtrlPred.dump_pred(ans)
            except tf.errors.OutOfRangeError:
                print "up to epoch limits"
            finally:
                coord.request_stop()
                coord.join(threads)


if __name__ == "__main__":
    FtrlPred.run()
