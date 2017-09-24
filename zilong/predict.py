from inputs import *
from network import *

import os


def predict_test_set():
    # read data use cpu, and train model use gpu
    with tf.device('/cpu:1'):
        # get all test files
        print "test_dir =", FLAGS.test_dir
        all_test_files = []
        test_walk = os.walk(FLAGS.test_dir)
        for path, dir, file_list in test_walk:
            for filename in file_list:
                if filename.startswith("part"):
                    all_test_files.append(os.path.join(path, filename))

        print "all_test_files_len", len(all_test_files)
        print "all_test_files", all_test_files

        # read test data
        filename_queue = tf.train.string_input_producer(tf.constant(all_test_files), num_epochs=1)
        test_batch = read_batch(filename_queue)

    # build test net
    _, test_predict, test_instance_id = inference_deep_wide(test_batch['deep_feature_index'],
                                                            test_batch['deep_feature_id'],
                                                            test_batch['wide_feature_index'],
                                                            test_batch['wide_feature_id'],
                                                            test_batch['instance_id'],
                                                            layers, 1)

    # global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # saver for reload training model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                           max_to_keep=100)

    # config
    graph_options = tf.GraphOptions(enable_bfloat16_sendrecv=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,
                                allow_growth=True)
    config = tf.ConfigProto(graph_options=graph_options,
                            gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        # read training model for test
        model_init = FLAGS.model_dir + "/" + FLAGS.model_name + '-' + str(FLAGS.max_steps)
        logger.info('model_init: %s', model_init)
        saver.restore(sess, model_init)
        logger.info('model restore done')

        predict_result_file = open(FLAGS.predict_out, "w")
        print >> predict_result_file, 'instance_id,prob'

        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            while True:
                step, instance_id, predict = sess.run([global_step, test_instance_id, test_predict])

                size_batch = len(instance_id)
                logger.info('batch size: %d', size_batch)
                for i in range(size_batch):
                    print >> predict_result_file, "%d,%f" % (instance_id[i][0], predict[i][0])
                    if step % 10000 == 0 and i == 0:
                        logger.info('instance_id: %d, predict: %f', instance_id[i][0], predict[i][0])

        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
            predict_result_file.close()
        logger.info('test finished')


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        predict_test_set()
