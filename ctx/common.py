# coding:utf-8
import os
import time
import sys
import random
import tensorflow as tf
from functools import partial
from ConfigParser import ConfigParser
import numpy as np
import math
from functools import partial


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
        self.model = cf.get(section, "model")

        self.training = tf.placeholder_with_default(False, shape=(), name='training')
        self.cf = cf

    def get_weight_size(self, vars):
        total_parameters = 0
        for var in vars:
            print var.name, var.shape
            variable_parameters = 1
            for dim in var.shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print "#params: %d" % total_parameters

    @staticmethod
    def leaky_relu(z, name=None):
        return tf.maximum(0.01 * z, z, name=name)

    def train_op(self, loss):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(
            self.lr, global_step, self.lr_decay_step,
            self.lr_decay_rate, staircase=True)

        vars = tf.trainable_variables()
        self.get_weight_size(vars)

        wide_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide')
        deep_vars = list(set(vars) - set(wide_vars))

        def max_norm_regularizer(weights, axes=1, name="max_norm", collection="max_norm"):
            # threshold = (16.0 / 256) ** 2
            threshold = 0.1
            clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
            clip_weights = tf.assign(weights, clipped, name=name)
            tf.add_to_collection(collection, clip_weights)

        for var in deep_vars:
            if "kernel" in var.name:
                max_norm_regularizer(var, name=var.name[:-2] + "_norm")

        if self.model == "WDE":
            ftrl = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=60.0)
            wide_opt = ftrl.minimize(loss, var_list=wide_vars)
        else:
            wide_opt = tf.no_op("wide_placehold")

        adam = tf.train.AdamOptimizer(learning_rate=lr)
        # adam = tf.train.AdagradOptimizer(learning_rate=lr)
        deep_opt = adam.minimize(loss, global_step=global_step, var_list=deep_vars)

        ema = tf.train.ExponentialMovingAverage(0.995, global_step)
        avg = ema.apply(tf.trainable_variables())
        return tf.group(deep_opt, wide_opt, avg)

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
