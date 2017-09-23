#!/usr/bin/python
import sys
import math

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

def read_file():
    labels = []
    probs = []
    for line in open(sys.argv[1]):
        sp = line.strip().split()
        try:
            label = int(sp[0])
            if label<=0:
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


def auc():
    import sys
    labels, probs = read_file()
    auc = scoreAUC(labels, probs)
    print("AUC  : %f" % auc)

def logloss():
    # outfile = open("loss", 'w')
    sum = 0.0
    count = 0

    for line in open(sys.argv[1]):    
        sp = line.strip().split()
        y = float(sp[0])
        p = float(sp[1])

        # factor = 0.1
        # if p < factor:
        #     p = factor
        # elif p > 1 - factor:
        #     p = 1 - factor
            
        sum += y * math.log(p) + (1-y)*math.log(1-p)
        count += 1
    ret = -sum/count
    print("logloss: %f" % ret)
    # outfile.write(str(ret) + "\n")
    
if __name__=="__main__":
    """usage : ./metrics.py filename"""
    auc()
    logloss()
