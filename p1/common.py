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
    def __init__(self, writer, step=100):
        self.aa = 0.0
        self.bb = 0.0
        self.writer = writer
        self.step = step

    def run(self, gs, loss, loss_valid):
        factor = 0.99
        if self.aa == 0.0:
            self.aa = loss
            self.bb = loss_valid
        else:
            self.aa = self.aa * factor + (1 - factor) * loss
            self.bb = self.bb * factor + (1 - factor) * loss_valid
        if gs % self.step == 0:
            out = "%s %5d %.3f %.3f %.3f" % (time.ctime(), gs, loss, self.aa, self.bb)
            print out
            print >> self.writer, out


def remove_opt_var(vars):
    ans = []
    print vars
    for _ in vars:
        if "Adag" in _.name or "Adam" in _.name:
            continue
        ans += [_]
    return ans


class NET(object):
    def __init__(self, cf):
        try:
            self.threshold = cf.getfloat("net", "threshold")
        except:
            self.threshold = 0.1
        self.step = 100
        section = "net"
        self.sparse_dim = cf.getint(section, "sparse_dim")
        self.layer_dim = eval(cf.get(section, "layer_dim"))
        try:
            self.hidden_factor = cf.getint(section, "hidden_factor")
        except:
            pass

        self.lr = cf.getfloat(section, "lr")
        self.lr_decay_step = cf.getint(section, "lr_decay_step")
        self.lr_decay_rate = cf.getfloat(section, "lr_decay_rate")
        self.model = cf.get(section, "model")

        self.training = tf.placeholder_with_default(False, shape=(), name='training')
        self.cf = cf

    def load_ftrl_weight(self, filename):
        weights = [0]
        for l in open(filename):
            l = l.strip()
            if not l:
                continue
            items = l.split()
            assert len(items) == 3
            w = float(items[2])
            weights += [w]
        assert len(weights) == self.sparse_dim, len(weights)
        weights = np.asarray(weights, dtype=np.float32).reshape((-1, 1))
        return weights

    @staticmethod
    def get_weight_size(self):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
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

        opts = []
        if wide_vars:
            print "wide_vars", wide_vars
            ftrl = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=0.0)
            wide_opt = ftrl.minimize(loss, var_list=wide_vars)
            opts += [wide_opt]

        if deep_vars:
            print "deep_vars", deep_vars
            adam = tf.train.AdamOptimizer(learning_rate=lr)
            # if self.model == "WDE":
            #     adam = tf.train.AdagradOptimizer(learning_rate=0.01)
            try:
                grads = adam.compute_gradients(loss, var_list=deep_vars)
                for i, (g, v) in enumerate(grads):
                    if g is not None:
                        grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
                deep_opt = adam.apply_gradients(grads, global_step=global_step)
                # deep_opt = adam.minimize(loss, global_step=global_step,
                opts += [deep_opt]
            except:
                pass

        print "ema,", self.ema_factor
        ema = tf.train.ExponentialMovingAverage(self.ema_factor, global_step)

        clip_all_weights = tf.get_collection("max_norm")

        with tf.control_dependencies(opts):
            with tf.control_dependencies(clip_all_weights):
                train_op = ema.apply(tf.trainable_variables())
        return train_op

    def loss_op(self, labels, logits):
        if labels.dtype != tf.float32:
            labels = tf.to_float(labels)
        xe = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        xe = tf.reduce_mean(xe)

        return xe


if __name__ == "__main__":
    pass
