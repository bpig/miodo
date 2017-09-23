# coding:utf-8
from common import *


class LoanData(object):
    def __init__(self, cf, is_pred):
        self.cf = cf
        if is_pred:
            section = "pred"
        else:
            section = "train"

        self.num_epochs = cf.getint(section, "num_epochs")
        self.batch_size = cf.getint(section, "batch_size")

        self.top_dir = cf.get(section, "top_dir")

        self.valid_dir = self.get_data_list("validate")
        self.train_dir = self.get_data_list("train")
        self.test_dir = self.get_data_list("test")
        
        self.is_train = not is_pred
        return

    def get_data_list(self, name, shuffle=False):
        dirname = self.top_dir + "/%s/" % name
        ans = [dirname + _ for _ in os.listdir(dirname) if _.startswith("part")]
        if shuffle:
            random.shuffle(ans)
        return ans
        

    def read(self, feature_map):
        if self.is_train:
            data_dir = self.train_dir
        else:
            data_dir = self.valid_dir
        assert len(data_dir)
        print len(data_dir)
        return self._read_by_queue(data_dir, self.num_epochs, self.batch_size, feature_map)

    def _read_by_queue(self, data_file, num_epochs, batch_size, feature_map):
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

    def read_valid(self, feature_map):
        return self._read_by_queue(self.valid_dir, None, self.batch_size, feature_map)


