# coding:utf-8
from common import *

if __name__ == "__main__":
    conf = sys.argv[1]
    conf = conf[:-5]
    filename = "log/%s_log/pred_result" % conf

    head = "instance_id,prob"
    fout = "log/%s_log/ans.csv" % conf
    print fout

    fout = open(fout, "w")
    print >> fout, head

    # 2163130
    ans = []
    for i, l in enumerate(open(filename)):
        l = l.strip()
        items = l.split()
        assert len(items) == 3
        ans += [[int(items[2]), items[1]]]

    ans = sorted(ans, key=lambda x:x[0])
    for key, score in ans:
        print >> fout, `key` + ',' + score
    print i
        
