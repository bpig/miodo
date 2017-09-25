from inputs import *
from network import *

import os


def predict_test_set():
  #read data use cpu, and train model use gpu
  with tf.device('/cpu:0'):
    ans = []
    for d in range(32, 33):
      prefix = "data/date=%2d/" % d
      ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    all_test_files = ans
    
    print "all_test_files", len(all_test_files)

    #read test data
    filename_queue = tf.train.string_input_producer(all_test_files, num_epochs=1)
    test_batch = read_batch(filename_queue, 0)

  #build test net
  _, test_predict, test_instance_id = inference_deep_wide(test_batch['deep_feature_index'],
                                                          test_batch['deep_feature_id'],
                                                          test_batch['wide_feature_index'],
                                                          test_batch['wide_feature_id'],
                                                          test_batch['instance_id'],
                                                          layers, 1)

  #global step
  global_step = tf.Variable(0, name='global_step', trainable=False)

  #saver for reload training model
  saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                         max_to_keep=100)

  #config
  gpu_options = tf.GPUOptions(allow_growth=True)
  config = tf.ConfigProto(gpu_options=gpu_options)
  model_out = "model/zl"
  
  with tf.Session(config=config) as sess:
    #read training model for test
    model_init = model_out + "-final"
    saver.restore(sess, model_init)

    predict_result_file = open("ans.raw", "w")

    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
      while True:
        step, instance_id, predict,label = sess.run([global_step, test_instance_id, test_predict, test_batch['label']])

        size_batch = len(instance_id)

        for i in range(size_batch):
          print >> predict_result_file, "%d %f %d" % (label[i][0], predict[i][0], instance_id[i][0])

    except tf.errors.OutOfRangeError as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  predict_test_set()
