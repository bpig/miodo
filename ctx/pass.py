#!/usr/bin/python
import collections
import sys
import math
import time
import os
from os.path import join, getsize
from collections import defaultdict
import numpy as np

a = [0.000387224, 0.01235583, 0.01974392, 0.03129793, 0.04737426, 0.06308955, 0.07734784, 0.09033871, 0.102949,
     0.115438, 0.127984, 0.140988, 0.154418, 0.168448, 0.1830732, 0.198823, 0.2165888, 0.238341, 0.268197, 0.316466, ]


def ss(score):
    for i in range(len(a)):
        if score < a[i]:
            return i
    return len(a)


def get_pair_by_pred(filename):
    for line in open(filename):
        line = line.strip()
        if not line:
            continue
        label, prob = map(float, line.split())
        yield label, prob


def get_distinct(filename):
    probs = []
    for label, prob in get_pair_by_pred(filename):
        probs += [prob]
    probs = np.asarray(probs)
    mi, mx = probs.min(), probs.max()
    print mi, mx
    dists = np.linspace(mi, mx, 10)
    print dists
    return dists


def get_dist_idx(dists, value):
    for i in range(len(dists) - 1):
        if value <= dists[i + 1]:
            return i
    assert False, value


def get_distinct_diff(dists, filename):
    dist_num = len(dists) - 1
    probs = [[] for _ in range(dist_num)]
    labels = [[] for _ in range(dist_num)]
    for label, prob in get_pair_by_pred(filename):
        idx = get_dist_idx(dists, prob)
        probs[idx] += [prob]
        labels[idx] += [label]
    factor = []
    for idx in range(dist_num):
        mean_prob = np.asarray(probs[idx]).mean()
        mean_label = np.asarray(labels[idx]).mean()
        f = mean_label / mean_prob
        factor += [f]
        print f, len(probs[idx])
    return factor


def process(dists, factor, filename):
    ori = 0.0
    sum = 0.0
    for ct, (label, prob) in enumerate(get_pair_by_pred(filename)):
        idx = get_dist_idx(dists, prob)
        ori += label * math.log(prob) + (1 - label) * math.log(1 - prob)
        prob *= factor[idx]
        sum += label * math.log(prob) + (1 - label) * math.log(1 - prob)

    sum = -sum / ct
    ori = -ori / ct
    print("logloss: %.6f, %.6f" % sum, ori)


if __name__ == "__main__":
    filename = sys.argv[1]
    if not os.path.exists(filename):
        conf = sys.argv[1]
        filename = "log/%s_log/pred_result" % conf

    mtime = time.ctime(os.stat(filename).st_mtime)
    filesize = "%.3fMB" % (getsize(filename) / 1024.0 / 1024.0)
    print mtime, filename, filesize

    dists = get_distinct(filename)
    factors = get_distinct_diff(dists, filename)
    process(dists, factors, filename)
