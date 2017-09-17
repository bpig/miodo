# coding:utf-8
# 2017/9/17 下午8:34
# 286287737@qq.com
from common import *


def dump():
    dirname = "/user/h_miui_ad/dev/wwxu/exp/tf/rfrecord/base/date=%2d"
    date = range(21, 34)
    prefix = "hadoop --cluster c3prc-hadoop fs -get "
    os.chdir("data")
    for d in date:
        cmd = prefix + dirname % d
        print time.ctime(), cmd


if __name__ == "__main__":
    dump()
    pass
