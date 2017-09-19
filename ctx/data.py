# coding:utf-8
# 2017/9/19 下午2:21
# 286287737@qq.com

from common import *


class Data(object):
    def __init__(self, cf, is_pred):
        self.cf = cf
        if is_pred:
            section = "pred"
        else:
            section = "train"
        try:
            self.data_file = cf.get("data_file", section)
            self.data_file = eval(self.data_file)
        except:
            self.data_file = None
        self.date_begin = cf.getint("date_begin", section)
        self.date_end = cf.getint("date_end", section)
        self.num_epochs = cf.getint("num_epochs", section)
        self.batch_size = cf.getint("batch_size", section)
        try:
            self.top_dir = cf.get("top_dir")
        except:
            self.top_dir = None

    def get_data_list(self):
        if self.data_file:
            return self.data_file
        ans = []
        for d in range(self.date_begin, self.date_end + 1):
            prefix = "%s/date=%2d/" % (self.top_dir, d)
            ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
        return ans

    def read(self):
        data_file = self.get_data_list()
        assert len(self.data_file)
        print len(self.data_file)
        filename_queue = tf.train.string_input_producer(
            data_file, num_epochs=self.num_epochs)
        reader = tf.TFRecordReader()
        key, value = reader.read(filename_queue)

        batch = tf.train.shuffle_batch(
            [value],
            batch_size=self.batch_size,
            num_threads=16,
            capacity=50000,
            min_after_dequeue=5000,
            allow_smaller_final_batch=False
        )

        return tf.parse_example(batch, features={
            'label': tf.FixedLenFeature([1], tf.int64),
            'fid': tf.VarLenFeature(tf.int64),
            'fval': tf.VarLenFeature(tf.int64),
            'iid': tf.FixedLenFeature(1, tf.int64),
        })


if __name__ == "__main__":
    pass
