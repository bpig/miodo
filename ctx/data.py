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

    def read(self, feature_map):
        self.get_data_list()
        assert len(self.data_file)
        print len(self.data_file)
        return self._read_by_queue(self.data_file, self.num_epochs, self.batch_size, feature_map)

    def _read_by_queue(self, data_file, num_epochs, batch_size, feature_map):
        filename_queue = tf.train.string_input_producer(
            data_file, num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        key, value = reader.read(filename_queue)

        batch = tf.train.batch(
            [value],
            batch_size=batch_size,
            num_threads=1,
            capacity=50000,
#            min_after_dequeue=5000,
            allow_smaller_final_batch=False
        )

        return tf.parse_example(batch, features=feature_map)

    def read_valid(self, feature_map):
        return self._read_by_queue(self.valid_data, None, self.batch_size, feature_map)


if __name__ == "__main__":
    def gen_embed(fea, sparse_dim, name):
        embed_dim = 2
        glorot = tf.uniform_unit_scaling_initializer
        weights = tf.get_variable("w_" + name, [sparse_dim, embed_dim],
                                  initializer=glorot)
        biases = tf.get_variable("b" + name, [embed_dim], initializer=tf.zeros_initializer)
        return tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="mean") + biases
    
    conf_file = sys.argv[1]
    cf = ConfigParser()
    cf.read("conf/" + conf_file)

    data = Data(cf, False)
    from multi_dnn import *

    fea = data.read(MultiDNN.feature_map)

    embeds = []
    keys = sparse_table.keys()
    with tf.variable_scope("embed"):
        for key in keys:
            v = fea[key + "_id"]
            embed = gen_embed(fea[key + "_id"], sparse_table[key], key)
            embeds += [embed, v, fea['iid']]
            
    # embed = tf.concat(embeds, axis=1)
    
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        cc = fea.values()
        print len(cc)
        d = 0
        ss = set()
        while not coord.should_stop():
        #with open("hah", "w"):
            a = sess.run(embeds)
            d += 1
            # print d
            for _ in range(0, len(a), 3):
                i = a[_]
                # print i.shape
                if i.shape[0] != 2:
                    # print a[_+1]
                    # print a[_+2]

                    l = len(ss)
                    ss.add(keys[_/3])
                    if len(ss) != l:
                        print keys[_/3]

                    # break
                # break
                #print i.indices.shape



