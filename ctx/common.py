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


class TrainLog():
    def __init__(self, writer):
        self.aa = 0.0
        self.bb = 0.0
        self.writer = writer

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
        print >> self.writer, out


class NET(object):
    def __init__(self, cf):
        section = "net"
        self.random_seed = 1314
        self.sparse_dim = cf.getint(section, "sparse_dim")
        self.layer_dim = eval(cf.get(section, "layer_dim"))
        self.batch_norm = False
        self.hidden_factor = cf.getint(section, "fm_factor")
        tf.set_random_seed(self.random_seed)

        self.lr = cf.getfloat(section, "lr")
        self.lr_decay_step = cf.getint(section, "lr_decay_step")
        self.lr_decay_rate = cf.getfloat(section, "lr_decay_rate")

    def get_weight_size(self):
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
            print variable.name, shape
        print "#params: %d" % total_parameters
        
    def train_op(self, loss):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(
            self.lr, global_step, self.lr_decay_step,
            self.lr_decay_rate, staircase=True)

        vars = tf.trainable_variables()
        # for var in vars:
        #     print var.name, var.shape
        wide_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide')
        deep_vars = list(set(vars) - set(wide_vars))

        # ftrl = tf.train.FtrlOptimizer(cf_float("ftrl_lr"), l1_regularization_strength=1.0)
        # wide_opt = ftrl.minimize(loss, var_list=wide_vars)

        # adam = tf.train.AdamOptimizer(learning_rate=lr)
        adam = tf.train.AdagradOptimizer(learning_rate=lr)
        deep_opt = adam.minimize(loss, global_step=global_step, var_list=deep_vars)

        ema = tf.train.ExponentialMovingAverage(0.99, global_step)
        avg = ema.apply(tf.trainable_variables())
        return tf.group(*[deep_opt, avg])

    def loss_op(self, labels, logits):
        if labels.dtype != tf.float32:
            labels = tf.to_float(labels)
        xe = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        xe = tf.reduce_mean(xe)
        # l2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # print l2
        # l2 = tf.reduce_mean(l2)
        return xe 


if __name__ == "__main__":
    pass
