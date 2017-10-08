#!/usr/bin/python
import collections
import sys
import math
import time
import os
from os.path import join, getsize
from collections import defaultdict
import numpy as np

def get_pair_by_pred(filename):
    for line in open(filename):
        line = line.strip()
        if not line:
            continue
        label, prob, _ = map(float, line.split())
        yield label, prob


def get_distinct(filename):
    probs = []
    for label, prob in get_pair_by_pred(filename):
        probs += [prob]
    probs = np.asarray(probs)
    mi, mx = probs.min(), probs.max()
    print mi, mx
    dists = np.linspace(mi, mx, 30)
    print dists
    return dists


# def get_distinct(filename):
#     probs = []
#     for label, prob in get_pair_by_pred(filename):
#         probs += [prob]
#     probs = np.asarray(probs)
#     mi, mx = probs.min(), probs.max()
#     print mi, mx
#     dists = []
#     for i in range(0, 100+1, 2):
#         dists += [np.percentile(probs, i)]
#     print dists
#     return dists


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
        try:
            mean_prob = np.asarray(probs[idx]).mean()
            mean_label = np.asarray(labels[idx]).mean()
            f = mean_label / mean_prob
        except:
            f = 0.0
        factor += [f]
        print idx, f, len(probs[idx])
    # print np.sum(np.sum(labels)) / np.sum(np.sum(probs))
    return factor


def process(dists, factor, filename):
    ori = 0.0
    cal = 0.0
    for ct, (label, prob) in enumerate(get_pair_by_pred(filename)):
        idx = get_dist_idx(dists, prob)
        if prob <= 0.001:
            prob = 0.001
        elif prob >= 0.999:
            prob = 0.999
        ori += label * math.log(prob) + (1 - label) * math.log(1 - prob)
        prob *= factor[idx]
        cal += label * math.log(prob) + (1 - label) * math.log(1 - prob)

    cal = -cal / ct
    ori = -ori / ct
    print("logloss: %.6f, %.6f" % (cal, ori))


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
