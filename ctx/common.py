# coding:utf-8
# 2017/9/16 下午11:00
# 286287737@qq.com
import os
import time
import sys
import tensorflow as tf
from functools import partial
from ConfigParser import ConfigParser


def read_config(filename):
    cf = ConfigParser()
    cf.read(filename)
    section = "basic"
    cf_str = partial(cf.get, section)
    cf_float = partial(cf.getfloat, section)
    cf_int = partial(cf.getint, section)
    return cf_str, cf_float, cf_int


if __name__ == "__main__":
    cf_str, _, _ = read_config("conf/1.conf")
    print cf_str("layers")
    pass
