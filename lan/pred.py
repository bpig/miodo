import xgboost as xgb
import sys, os

bst = xgb.Booster({'nthread':32}) 
bst.load_model(sys.argv[1]) 

dte = xgb.DMatrix("data/pred.fea") 
ypred = bst.predict(dte)

labels = open("data/pred.lb").readlines()
labels = [_.strip() for _ in labels]
sy = sorted(ypred)
print sy[:10]
print sy[-10:]

fout = open("ans.csv", "w")
print >> fout, "user_id,ovd_rate"
for label, prob in zip(labels, ypred):
    print >> fout, label + ",%.6f" % prob
