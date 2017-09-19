# coding:utf-8
# 2017/9/16 下午11:00
# 286287737@qq.com
import os
import time
import sys
import tensorflow as tf
from functools import partial
from ConfigParser import ConfigParser
import numpy as np


class NET(object):
    def __init__(self, cf):
        section = "net"
        self.random_seed = 1314
        self.sparse_dim = cf.getint(section, "sparse_dim")
        self.layer_dim = eval(cf.get(section, "layer_dim"))
        self.batch_norm = False
        self.hidden_factor = cf.getint(section, "fm_factor")
        tf.set_random_seed(self.random_seed)

        self.lr = cf.get_float(section, "lr")
        self.lr_decay_step = cf.getint(section, "lr_decay_step")
        self.lr_decay_rate = cf.getfloat(section, "lr_decay_rate")

    def train_op(self, loss):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(
            self.lr, global_step, self.lr_decay_step,
            self.lr_decay_rate, staircase=True)

        vars = tf.trainable_variables()
        wide_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide')
        deep_vars = list(set(vars) - set(wide_vars))

        # ftrl = tf.train.FtrlOptimizer(cf_float("ftrl_lr"), l1_regularization_strength=1.0)
        # wide_opt = ftrl.minimize(loss, var_list=wide_vars)

        adam = tf.train.AdamOptimizer(learning_rate=lr)
        deep_opt = adam.minimize(loss, global_step=global_step, var_list=deep_vars)

        ema = tf.train.ExponentialMovingAverage(0.99, global_step)
        avg = ema.apply(tf.trainable_variables())
        return tf.group(*[deep_opt, avg])

    def loss_op(labels, logits):
        xe = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(xe)


if __name__ == "__main__":
    pass
