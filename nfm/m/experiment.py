import time

from inputs import *
from network import *
from metrics import calc_deep_wide_metrics

import os, random


def train():
  #read data use cpu, and train model use gpu
  with tf.device('/cpu:0'):
    #get all train files
    print "training_dir =", FLAGS.train_dir
    all_train_files = []
    train_walk = os.walk(FLAGS.train_dir)
    for path, dir, file_list in train_walk:
      for filename in file_list:
        if filename.startswith("part"):
          all_train_files.append(os.path.join(path, filename))
    all_train_files = sorted(all_train_files)

    print "all_train_files_len", len(all_train_files)
    print "all_train_files", all_train_files

    if FLAGS.is_shuffle:
      random.shuffle(all_train_files)

    #get all valid files
    print "validation_dir =", FLAGS.valid_dir
    all_valid_files = []
    valid_walk = os.walk(FLAGS.valid_dir)
    for path, dir, file_list in valid_walk:
      for filename in file_list:
        if filename.startswith("part"):
          all_valid_files.append(os.path.join(path, filename))

    print "all_valid_files_len", len(all_valid_files)
    print "all_valid_files", all_valid_files

    #read training data
    training_fq = tf.train.string_input_producer(tf.constant(all_train_files))
    training_batch = read_batch(training_fq, FLAGS.is_shuffle)

    #read validation data
    validation_fq = tf.train.string_input_producer(tf.constant(all_valid_files), num_epochs=1)
    validation_batch = read_all_batch(validation_fq)

  #build training net
  training_logits, _, _ = inference_deep_wide(training_batch['deep_feature_index'],
                                              training_batch['deep_feature_id'],
                                              training_batch['wide_feature_index'],
                                              training_batch['wide_feature_id'],
                                              training_batch['instance_id'],
                                              layers, 0.5)

  #build loss function
  training_sum_loss, training_mean_loss = log_loss(training_logits,
                                                   training_batch['label'])
  tf.summary.scalar('training_mean_loss', training_mean_loss)

  #get all variables
  all_var = tf.global_variables()
  for var in all_var:
    print(var, var.name)
  print('all_var', len(all_var))

  #get all trainable variables
  trainable_var = tf.trainable_variables()
  for var in trainable_var:
    print(var, var.name, var.get_shape())
  print 'trainable_var', len(trainable_var)

  #get all trainable wide variables
  trainable_wide_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_hidden1')
  for var in trainable_wide_var:
    print(var, var.name, var.get_shape())
  print 'trainable_wide_var', len(trainable_wide_var)

  #get all trainable deep variables
  #trainable_deep_var = []
  #for var in trainable_var:
  #  if var not in trainable_wide_var:
  #    trainable_deep_var.append(var)
  #    print(var, var.name, var.get_shape())
  #print 'trainable_deep_var', len(trainable_deep_var)

  #global step
  global_step = tf.Variable(0, name='global_step', trainable=False)
  #deep training optimizer, update the net parameters
  ada_optimizer = tf.train.AdagradOptimizer(0.01)
  adam_optimizer = tf.train.AdamOptimizer(0.01)
  #deep_train_op = ada_optimizer.minimize(training_mean_loss,
  #                                       global_step=global_step,
  #                                       var_list=trainable_deep_var)

  #wide training optimizer, update the net parameters
  ftrl_optimizer = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=10.0)
  wide_train_op = ftrl_optimizer.minimize(training_mean_loss,
                                          var_list=trainable_wide_var)

  #training deep and wide variables together
  #train_op = tf.group(deep_train_op, wide_train_op)
  train_op = wide_train_op

  #init all variables together
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  summary_op = tf.summary.merge_all()

  #writer for output log
  writer = tf.summary.FileWriter(FLAGS.log_dir,
                                 tf.get_default_graph())

  #saver for saving training model
  saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                         max_to_keep=100)

  #session config
  graph_options = tf.GraphOptions(enable_bfloat16_sendrecv=True)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95,
                              allow_growth=True)
  config = tf.ConfigProto(graph_options=graph_options,
                          gpu_options=gpu_options,
                          log_device_placement=False,
                          allow_soft_placement=True)

  with tf.Session(config=config) as sess:
    #init operation
    sess.run(init_op)
    #training multi threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    #model saved path
    model_out = FLAGS.model_dir + "/" + FLAGS.model_name
    print "model_out =", model_out
    total_training = 0
    step = 0
    try:
      while not coord.should_stop():
        start = time.time()

        #the data or operation to be update and get output
        _, sum_loss, mean_loss, summary = \
          sess.run([train_op,
                    training_sum_loss,
                    training_mean_loss,
                    summary_op])

        end = time.time()
        total_training += end - start

        step += 1
        #print log
        if step % FLAGS.log_per_batch == 0:
          writer.add_summary(summary, step)
          logger.info('step: %09d, training_sum_loss: %f, training_mean_loss: %f', step, sum_loss, mean_loss)

        #save model
        if step % FLAGS.save_per_batch == 0:
          saver.save(sess, model_out, global_step=step)

        #validation
        if step % FLAGS.valid_per_batch == 0:
          start = time.time()

          #validation metrics
          validation_sum_loss, validation_mean_loss, validation_auc = calc_deep_wide_metrics(sess, validation_batch)

          tf.summary.scalar('validation_sum_loss', validation_sum_loss)
          tf.summary.scalar('validation_mean_loss', validation_mean_loss)
          tf.summary.scalar('validation_auc', validation_auc)

          end = time.time()
          total_validation = end - start
          logger.info('step: %09d, training_time: %f, validation_time: %f, validation_sum_loss: %f, validation_mean_loss: %f, validation_auc: %f',
                      step,
                      total_training,
                      total_validation,
                      validation_sum_loss,
                      validation_mean_loss,
                      validation_auc)

          total_training = 0

        #finish training
        if step >= FLAGS.max_steps:
          logger.info('training finished with step: %09d, max_steps: %09d', step, FLAGS.max_steps)
          break
    except tf.errors.OutOfRangeError as e:
      coord.request_stop(e)
    finally:
      saver.save(sess, model_out, global_step=FLAGS.max_steps)
      #stop all threads
      coord.request_stop()
      coord.join(threads)
      writer.close()

if __name__ == '__main__':
  with tf.device('/gpu:0'):
    train()
