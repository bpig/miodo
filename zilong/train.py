import time

from inputs import *
from network import *

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
        out = "%5d %.3f %.3f %.3f" % (gs, loss, self.aa, self.bb)
        print out

def get_data_list(top_dir, begin, end):
    ans = []
    for d in range(begin, end + 1):
        prefix = "%s/date=%2d/" % (top_dir, d)
        ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    random.shuffle(ans)
    return ans


def train():
    top_dir = "dw/"
    train_data = get_data_list(top_dir, 24, 27)
    valid_data = get_data_list(top_dir, 29, 30)
    print len(train_data)
    print len(valid_data)    

    fq = tf.train.string_input_producer(train_data)
    fea = read_batch(fq)

    fq = tf.train.string_input_producer(valid_data, num_epochs=1)
    valid = read_batch(fq)

    logits = inference_deep_wide(fea)

    loss = loss_op(logits, fea['label'])

    global_step = tf.train.get_or_create_global_step()    

    vars = tf.trainable_variables()
    for var in vars:
        print var.name, var.get_shape()

    wide_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide')
    for var in wide_vars:
        print var, var.name, var.get_shape()

    deep_vars = list(set(vars) - set(wide_vars))



    ada_optimizer = tf.train.AdagradOptimizer(0.01)
    # adam_optimizer = tf.train.AdamOptimizer(0.01)
    deep_train_op = ada_optimizer.minimize(
        loss, global_step=global_step, var_list=deep_vars)

    ftrl_optimizer = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=20.0)
    wide_train_op = ftrl_optimizer.minimize(
        loss, var_list=wide_vars)

    train_op = tf.group(deep_train_op, wide_train_op)

    tf.get_variable_scope().reuse_variables()
    logits2 = inference_deep_wide(valid)
    loss2 = loss_op(logits2, valid['label'])
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        log = TrainLog()
        model_out = "model/zilong"
        os.system("mkdir %s" % model_out)
        try:
            while not coord.should_stop():
                _, v_loss, t_loss, gs = sess.run([train_op, loss2, loss, global_step])
                log.run(gs, t_loss, v_loss)

                if gs % 500 == 0:
                    saver.save(sess, model_out, global_step=gs)

                if gs >= 50000:
                    print 'break for max_steps: %09d', gs
                    break
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            saver.save(sess, model_out + "-final")
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()
