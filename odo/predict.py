from data import *
from network import *

import os


def pred():
    fea = read_pred()
    layers = eval("[%s]" % FLAGS.layers)
    _, pred, iid = inference_deep_wide(fea, layers, 1)

    global_step = tf.train.get_or_create_global_step()

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    top_dir = "model/%d/" % FLAGS.model
    model_path = top_dir + "m%s-%s" % (FLAGS.model, FLAGS.model_version)
    print model_path
    ans = top_dir + "ans.raw"

    with tf.Session(config=config) as sess:
        restore_model(sess, model_path)
        sess.run(tf.local_variables_initializer())
        fout = open(ans, "w")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            while True:
                step, imei, prob, label = sess.run(
                    [global_step, iid, pred, fea['label']])

                for i in range(len(imei)):
                    print >> fout, "%d %f %d" % (label[i][0], prob[i][0], imei[i][0])

        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    pred()
