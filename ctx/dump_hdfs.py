# coding:utf-8
# 2017/9/17 下午8:34
# 286287737@qq.com
from common import *


def dump():
    # dirname = "/user/h_miui_ad/dev/wwxu/exp/tf/rfrecord/base/date=%2d"
    dirname = "/user/h_miui_ad/dev/wwxu/exp/tf/rfrecord/opt5_1/date=%2d"
    date = range(20, 31)
    prefix = "hadoop --cluster c3prc-hadoop fs -get "
    top_dir = dirname.split("/")[-2]
    print top_dir
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)
    os.chdir(top_dir)
    for d in date:
        cmd = prefix + dirname % d
        print time.ctime(), cmd
        ret = os.system(cmd)
        print ret


if __name__ == "__main__":
    dump()
    pass
