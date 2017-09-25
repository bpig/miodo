import time

from inputs import *
from network import *
from metrics import calc_deep_wide_metrics

import os, random

class TrainLog():
    def __init__(self):
        self.aa = 0.0
        self.bb = 0.0

    def run(self, gs, loss, loss_valid):
        factor = 0.99
        if self.aa == 0.0:
            self.aa = loss
            self.bb = loss_valid
        else:
            self.aa = self.aa * factor + (1 - factor) * loss
            self.bb = self.bb * factor + (1 - factor) * loss_valid
        out = "%d %.3f %.3f %.3f" % (gs, loss, self.aa, self.bb)
        print out

def train():

  with tf.device('/cpu:0'):
    ans = []
    for d in range(11, 31):
      prefix = "data/date=%2d/" % d
      ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    random.shuffle(ans)
    
    all_train_files = ans

    print "all_train_files_len", len(all_train_files)

    training_fq = tf.train.string_input_producer(all_train_files)
    training_batch = read_batch(training_fq, FLAGS.is_shuffle)

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

  trainable_wide_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_hidden1')
  for var in trainable_wide_var:
    print(var, var.name, var.get_shape())
  print 'trainable_wide_var', len(trainable_wide_var)

  trainable_deep_var = []
  for var in trainable_var:
    if var not in trainable_wide_var:
      trainable_deep_var.append(var)
      print(var, var.name, var.get_shape())
  print 'trainable_deep_var', len(trainable_deep_var)


  global_step = tf.Variable(0, name='global_step', trainable=False)

  ada_optimizer = tf.train.AdagradOptimizer(0.01)
  deep_train_op = ada_optimizer.minimize(training_mean_loss,
                                         global_step=global_step,
                                         var_list=trainable_deep_var)


  ftrl_optimizer = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=1.0)
  wide_train_op = ftrl_optimizer.minimize(training_mean_loss,
                                          var_list=trainable_wide_var)

  train_op = tf.group(deep_train_op, wide_train_op)
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

  saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10)

  gpu_options = tf.GPUOptions(allow_growth=True)
  config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.Session(config=config) as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    model_out = "model/zl"
    print "model_out =", model_out
    total_training = 0
    ll = TrainLog()
    try:
      while not coord.should_stop():
        start = time.time()

        #the data or operation to be update and get output
        _, sum_loss, mean_loss, step = \
          sess.run([train_op,
                    training_sum_loss,
                    training_mean_loss,
                    global_step])

        end = time.time()
        total_training += end - start
        
        if step % FLAGS.log_per_batch == 0:
          ll.run(step, sum_loss, mean_loss)
          logger.info('step: %09d, training_sum_loss: %f, training_mean_loss: %f', step, sum_loss, mean_loss)

        #save model
        if step % FLAGS.save_per_batch == 0:
          saver.save(sess, model_out, global_step=step)

        if step >= FLAGS.max_steps:
          logger.info('training finished with step: %09d, max_steps: %09d', step, FLAGS.max_steps)
          break
    except tf.errors.OutOfRangeError as e:
      pass
    finally:
      saver.save(sess, model_out + "-final")
      #stop all threads
      coord.request_stop()
      coord.join(threads)
      writer.close()

if __name__ == '__main__':
  train()
