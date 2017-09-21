# coding:utf-8
from common import *


def read(num_epochs=1):
    data_filename = "part-r-00000"
    filename_queue = tf.train.string_input_producer(
        [data_filename], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    return value


if __name__ == "__main__":
    value = read()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        c = 0
        try:
            while not coord.should_stop():
                sess.run(value)
                if c % 1000 == 0:
                    print c
                c += 1
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            coord.request_stop()
            coord.join(threads)
            print c
    pass
