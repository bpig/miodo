#!/usr/bin/python
import sys
import math
import time
import os
from os.path import join, getsize


def scoreAUC(labels, probs):
    i_sorted = sorted(xrange(len(probs)), key=lambda i: probs[i],
                      reverse=True)
    auc_temp = 0.0
    TP = 0.0
    TP_pre = 0.0
    FP = 0.0
    FP_pre = 0.0
    P = 0;
    N = 0;
    last_prob = probs[i_sorted[0]] + 1.0
    for i in xrange(len(probs)):
        if last_prob != probs[i_sorted[i]]:
            auc_temp += (TP + TP_pre) * (FP - FP_pre) / 2.0
            TP_pre = TP
            FP_pre = FP
            last_prob = probs[i_sorted[i]]
        if labels[i_sorted[i]] == 1:
            TP = TP + 1
        else:
            FP = FP + 1
    auc_temp += (TP + TP_pre) * (FP - FP_pre) / 2.0
    auc = auc_temp / (TP * FP)
    return auc


def read_file(filename):
    labels = []
    probs = []
    for line in open(filename):
        sp = line.strip().split()
        try:
            label = int(sp[0])
            if label <= 0:
                label = 0
            else:
                label = 1
            prob = float(sp[1])
        except:
            print line
            continue
        labels.append(label)
        probs.append(prob)
    return (labels, probs)


def auc(filename):
    labels, probs = read_file(filename)
    auc = scoreAUC(labels, probs)
    print("AUC  : %f" % auc)


def logloss(filename):
    sum = 0.0
    count = 0

    for line in open(filename):
        sp = line.strip().split()
        y = float(sp[0])
        p = float(sp[1])

        if 0.5 <= p < 0.6:
            p = 0.5

        if p < 0.001:
            p *= 1.1

        # elif p > 0.983:
        #     p = 0.983

        sum += y * math.log(p) + (1 - y) * math.log(1 - p)
        count += 1
    ret = -sum / count
    print("logloss: %f" % ret)
    # outfile.write(str(ret) + "\n")


if __name__ == "__main__":
    """usage : ./metrics.py filename"""

    filename = sys.argv[1]
    if not os.path.exists(filename):
        conf = sys.argv[1]
        filename = "log/%s_log/pred_result" % conf

    mtime = time.ctime(os.stat(filename).st_mtime)
    filesize = "%.3fMB" % (getsize(filename) / 1024.0 / 1024.0)
    print mtime, filename, filesize

    auc(filename)
    logloss(filename)
