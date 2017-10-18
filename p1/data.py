# coding:utf-8
from common import *


class Data(object):
    def __init__(self, cf, is_pred):
        self.cf = cf
        if is_pred:
            section = "pred"
        else:
            section = "train"

        self.date_begin = cf.getint(section, "date_begin")
        self.date_end = cf.getint(section, "date_end")
        self.num_epochs = cf.getint(section, "num_epochs")
        self.batch_size = cf.getint(section, "batch_size")

        self.top_dir = cf.get(section, "top_dir")
        if section == "train":
            self.valid_data = []
            for d in [29, 30]:
                prefix = "%s/date=%2d/" % (self.top_dir, d)
                print prefix
                self.valid_data += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
            print "valid", len(self.valid_data)

    def get_data_list(self):
        ans = []
        for d in range(self.date_begin, self.date_end + 1):
            prefix = "%s/date=%2d/" % (self.top_dir, d)
            print prefix
            ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
        random.shuffle(ans)
        self.data_file = ans
        return

    def read(self, feature_map):
        self.get_data_list()
        assert len(self.data_file)
        print len(self.data_file)
        return self._read_by_queue(self.data_file, self.num_epochs, self.batch_size, feature_map)

    def _read_by_queue(self, data_file, num_epochs, batch_size, feature_map, name="train"):
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

    def read_valid(self, feature_map):
        return self._read_by_queue(self.valid_data, None, self.batch_size, feature_map, "valid")


if __name__ == "__main__":
    tf.nn.elu()
    pass
