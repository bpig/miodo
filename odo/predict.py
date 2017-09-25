from data import *
from network import *

import os


def predict_test_set():
    fea = read_pred()
    layers = eval("[%s]" % FLAGS.layers)
    _, pred, iid = inference_deep_wide(
        fea['deep_feature_id'], fea['wide_feature_id'], fea['instance_id'], layers, 1)

    global_step = tf.train.get_or_create_global_step()

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    FLAGS.model = "model/%d/m%d" % (FLAGS.model, FLAGS.model)
    model_path = FLAGS.model + "-" + FLAGS.model_version
    print model_path

    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)

        fout = open(FLAGS.ans, "w")

        sess.run(tf.local_variables_initializer())
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
    predict_test_set()
