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
            auc_temp += (TP+TP_pre) * (FP-FP_pre) / 2.0        
            TP_pre = TP
            FP_pre = FP
            last_prob = probs[i_sorted[i]]
        if labels[i_sorted[i]] == 1:
          TP = TP + 1
        else:
          FP = FP + 1
    auc_temp += (TP+TP_pre) * (FP-FP_pre) / 2.0
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
                label=0
            else:
                label=1
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

a = [0.000387224 , 0.01235583 , 0.01974392 , 0.03129793 , 0.04737426 , 0.06308955 , 0.07734784 , 0.09033871 , 0.102949 , 0.115438 , 0.127984 , 0.140988 , 0.154418 , 0.168448 , 0.1830732 , 0.198823 , 0.2165888 , 0.238341 , 0.268197 , 0.316466 ,]
def ss(score):
    
    for i in range(len(a)):
        if score < a[i]:
            return i
    return len(a)
    
def logloss(filename):
    sum = 0.0
    count = 0
    v = []
    ll = [[0.0, 0.0] for _ in range(len(a) + 1)]
    bb = [[0.0, 0.0] for _ in range(len(a) + 1)]
    vv = [0.641780347774 , 0.662413461274 , 0.779448119186 , 0.932038259017 , 1.01701845969 , 1.03657146435 , 1.05397333451 , 1.07031761977 , 1.06932969788 , 1.07253644402 , 1.06562320169 , 1.05370565878 , 1.06147968757 , 1.05749828458 , 1.05275659056 , 1.06076309508 , 1.07788049236 , 1.0588050606 , 1.05026210902 , 1.01708013425 ,]
    pp = 0
    yy = 0.0
    for line in open(filename):
        sp = line.strip().split()
        y = float(sp[0])
        p = float(sp[1])
        # p = p * float(sys.argv[2])
        v += [p]
        idx = ss(p)
        pp += p
        yy += y
        
        if idx >= 1:
            # p *= float(sys.argv[2])
            p *= vv[idx - 1]
        # if p < 0.001:
        #     p = 0.001
        # elif p > 1.0:
        #     p = 0.999
        bb[idx][0] += y
        bb[idx][1] += p
        ls = y * math.log(p) + (1-y)*math.log(1-p)
        ll[idx][0] += ls
        ll[idx][1] += 1
        sum += ls
        count += 1
    print ll
    ret = -sum/count
    import numpy as np
    v = np.asarray(v)
    for i in range(0, 100, 5):
        d = np.percentile(v, i)
        print d ,",",
    print
    oo = [-0.0345742644645 , -0.0593076866905 , -0.0964969516165 , -0.156557263578 , -0.216371287076 , -0.261003881419 , -0.298894546078 , -0.332724406086 , -0.360703918704 , -0.387610187914 , -0.411121601021 , -0.432551812368 , -0.458242209602 , -0.480361843805 , -0.501917636006 , -0.527368333356 , -0.55712143049 , -0.580934846255 , -0.614613449917 , -0.663926189475 , ]
    for i in range(1, len(ll)):
        print i, a[i-1], bb[i][0] / ll[i][1], bb[i][1]/ll[i][1], ll[i][0] / ll[i][1], oo[i-1],  -(ll[i][0] / ll[i][1] -  oo[i-1])
        # print ll[i][0] / ll[i][1], ",",
    for i in range(1, len(ll)):
        print bb[i][0] / ll[i][1] /( bb[i][1]/ll[i][1]), ",",
    print
    print pp / count, yy / count, yy / pp
    print("logloss: %f, 0.371621" % ret)
    # outfile.write(str(ret) + "\n")
    
if __name__=="__main__":
    """usage : ./metrics.py filename"""

    filename = sys.argv[1]
    if not os.path.exists(filename):
        conf = sys.argv[1]
        filename = "log/%s_log/pred_result" % conf
    # 
    mtime  = time.ctime(os.stat(filename).st_mtime)
    filesize = "%.3fMB" % (getsize(filename) / 1024.0 / 1024.0)
    print mtime, filename, filesize

    # auc(filename)
    logloss(filename)
