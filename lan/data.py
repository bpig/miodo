import xgboost as xgb

s = open('loan/ee').readline().split()


dtrain = xgb.DMatrix(

print dtrain
