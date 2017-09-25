#coding:utf-8
# http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.core
# https://github.com/dmlc/xgboost/blob/master/doc/python/python_intro.md

from scipy.sparse import coo_matrix
import xgboost as xgb
import os
import sys
import numpy as np
import time

def get_data(dirname):
    dirname += "/"
    files = [dirname + _ for _ in os.listdir(dirname) if _.startswith("part")]
    print files
    for f in files:
        print time.ctime(), f
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            items = l.split()
            label = int(items[0])
            feas = [_.split(":") for _ in items[1:]]
            fid, value = zip(*feas)
            assert len(fid) == len(value)
            value = map(float, value)
            yield label, fid, value
            

if __name__ == "__main__":
    dirname = sys.argv[1]
    data = []
    row = []
    col = []
    labels = []
    print "======== load ==========="
    print time.ctime(), "begin"        
    for i, (label, fid, fval) in enumerate(get_data(dirname)):
        fid = [int(_.split("_")[1]) for _ in fid]
        
        ct = len(fid)
        row += [[i] * ct]
        col += [fid]
        data += [fval]
        labels += [label]
    print time.ctime(), "end"            

    print "======== dump ==========="    
    print time.ctime(), "begin"    
        
    data = np.concatenate(data)
    row = np.concatenate(row)
    col = np.concatenate(col)
    coo = coo_matrix((data, (row, col)))
    print coo.shape

    dm = xgb.DMatrix(coo, labels)
    print dm.num_row(), dm.num_col()
    print dm.get_label()

    dm.save_binary("dm.xgb")
    print time.ctime(), "finish"

    print "======== split ==========="
    print time.ctime(), "begin"
    row = dm.num_row()
    ct = int(row * 0.95)
    print ct, row
    row = range(row)
    random.shuffle(row)
    
    train = dm.slice(row[:ct])
    test = dm.slice(row[ct:])

    train.save_binary("tr.xgb")
    test.save_binary("te.xgb")    
    print time.ctime(), "finish"
