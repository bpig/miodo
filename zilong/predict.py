from inputs import *
from network import *

import os
import random


def get_data_list(top_dir, begin, end):
    ans = []
    for d in range(begin, end + 1):
        prefix = "%s/date=%2d/" % (top_dir, d)
        ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    random.shuffle(ans)
    return ans


def pred():
    top_dir = "dw/"
    valid_data = get_data_list(top_dir, 29, 30)

    filename_queue = tf.train.string_input_producer(valid_data, num_epochs=1)
    batch = read_batch(filename_queue)

    logits = inference_deep_wide(batch)
    prob = tf.sigmoid(logits)

    saver = tf.train.Saver()

    # graph_options = tf.GraphOptions(enable_bfloat16_sendrecv=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    model_out = "model/zilong-final"

    with tf.Session(config=config) as sess:
        saver.restore(sess, model_out)

        fout = open("pred_result", "w")

        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while True:
                lb, iid, pre = sess.run([batch['label'], batch['iid'], prob])
                for i in range(len(iid):
                    print >> fout, "%d %f %s" % (lb[i][0], pre[i][0], iid[i][0])

        except tf.errors.OutOfRangeError as e:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    pred()
