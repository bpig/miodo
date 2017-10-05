import os
import sys

dirname = sys.argv[1]
tr = "train"
te = "test"

os.chdir(dirname)

os.system("mv %s date=28" % tr)
os.system("mv %s date=29" % te)
os.mkdir("date=30")
# os.system("mv date=28/part-r-00000 date=29/")

