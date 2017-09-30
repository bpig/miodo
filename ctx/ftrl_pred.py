# coding:utf-8
# 2017/9/30 下午1:15
# 286287737@qq.com
from common import *


class FtrlPred:
    @staticmethod
    def get_data_list():
        ans = []
        top_dir = "/home/work/wwxu/opt1_100/"
        for d in range(29, 30):
            prefix = "%s/date=%2d/" % (top_dir, d)
            print prefix
            ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
        # random.shuffle(ans)
        return ans

    @staticmethod
    def read(feature_map):
        ans = FtrlPred.get_data_list()[:1]
        # ans = ['/home/work/wwxu/27/opt1_100//date=30/part-r-00370']
        print len(ans)
        # print ans
        return FtrlPred._read_by_queue(ans, 1, 1, feature_map)

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
        weights = [0]  # * self.sparse_dim
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

        weights = np.asarray(weights, dtype=np.float32).reshape((-1))
        print weights.shape
        # a = "0 0.311975"
        # ids = "13479 13750 21795 22113 19 2105 6859 17386 4549 12686 10443 16963 17561 18234 8173 17015 21614 11307 22412 18620 15019 11941 10454 14356 1764 19708 8482 2804 26610 21641 244 2054 9029 13645 28014 9635 8633 1768 14202 4008 21361 26118 19343 14919 23608 25384 24395 10447 25383 14214 8096 23458 6040 5910 25462 5823 51 23829 14658 13667 16968 21941".split()
        # 0 0.208664 290048846
        ids = "22028 10383 19 15593 2516 4380 23838 8537 20841 17039 4675 8932 25510 9010 17015 14868 27634 22019 24173 1183 14478 54 10365 3524 22118 8662 4176 5666 25838 12737 2233 22223 12320 10749 3681 423 10595 3704 1832 27628 11276 21352 10805 14919 16765 22428 19995 28050 24395 3609 7759 21277 25252 9669 24008 26484 27703 805 6719 4556 3164 22127 1772 9751 19094 23829 25846 8526 11963 14685 432".split()
        ids = map(int ,ids)

        logit =  weights[ids].sum()- 0.614403
        print logit
        logit = 0.0
        for i in ids:
            logit += weights[i]
        logit -= 0.614403
        print logit 
        FtrlPred.sigmoid(logit)
        return weights
    @staticmethod
    def sigmoid(x):
        import math
        print 1.0 / (1 + pow(math.e, -x))
    
    @staticmethod
    def inference(fea):
        fea = fea['fid']

        with tf.variable_scope("ftrl"):
            ftrl_weight = FtrlPred.load_ftrl_weight("fea2id_opt1_100")
            weights = tf.Variable(ftrl_weight, name="ftrl_weight", trainable=False)
            v = fea.values
            # fea = tf.sparse_to_indicator(fea, 28677)
            # fea = tf.to_float(fea)
            # ftrl = tf.matmul(fea, weights) - 0.614403

            ftrl = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="sum") - 0.614403
        return tf.sigmoid(ftrl), v

    @staticmethod
    def dump_pred(ans, fout):
        label, prob, iid, vv = [_.reshape(-1) for _ in ans]
        # print ans
        # for v1, v2, v3, v4 in zip(label, prob, iid, vv):
            # if v3 == 290737151:
            #     print v1, v2, v3
        print >> fout, label[0], prob[0], iid[0], " ".join(map(str, vv))

    @staticmethod
    def run():
        kv = FtrlPred.read(FtrlPred.feature_map)
        pred, v = FtrlPred.inference(kv)
        gpu_options = tf.GPUOptions(allow_growth=True)

        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            fout = open("www", "w")
        
            try:
                while not coord.should_stop():
                    ans = sess.run([kv['label'], pred, kv['iid'], v])
                    FtrlPred.dump_pred(ans, fout)
            except tf.errors.OutOfRangeError:
                print "up to epoch limits"
            finally:
                coord.request_stop()
                coord.join(threads)


if __name__ == "__main__":
    # FtrlPred.run()
    FtrlPred.load_ftrl_weight("fea2id_opt1_100")
