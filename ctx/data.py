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
            self.data_file = cf.get(section, "data_file")
            self.data_file = eval(self.data_file)
        except:
            self.data_file = None
        self.date_begin = cf.getint(section, "date_begin")
        self.date_end = cf.getint(section, "date_end")
        self.num_epochs = cf.getint(section, "num_epochs")
        self.batch_size = cf.getint(section, "batch_size")

        if section == "train":
            self.valid_data = eval(cf.get(section, "valid_data"))
            self.top_dir = cf.get(section, "top_dir")

    def get_data_list(self):
        if self.data_file:
            return
        ans = []
        for d in range(self.date_begin, self.date_end + 1):
            prefix = "%s/date=%2d/" % (self.top_dir, d)
            ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
        self.data_file = ans
        return

    def read(self):
        self.get_data_list()
        assert len(self.data_file)
        print len(self.data_file)
        return self._read_by_queue(self.data_file, self.num_epochs, self.batch_size)

    def _read_by_queue(self, data_file, num_epochs, batch_size):
        filename_queue = tf.train.string_input_producer(
            data_file, num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        key, value = reader.read(filename_queue)

        batch = tf.train.shuffle_batch(
            [value],
            batch_size=batch_size,
            num_threads=8,
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

    def read_valid(self):
        return self._read_by_queue(self.valid_data, self.num_epochs, self.batch_size)

    if __name__ == "__main__":
        pass
