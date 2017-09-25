# coding:utf-8
# http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.core
# https://github.com/dmlc/xgboost/blob/master/doc/python/python_intro.md

from common import *

def load_ids_map():
    filename = "v1/ids.map"
    def parse(x):
        x = x.strip()
        k, v = x.split()
        v = int(v)
        return k, v
    kv = map(parse, open(filename).readlines())
    print len(kv)
    return dict(kv)


def ids(key):
    global kv
    if key not in kv:
        kv[key] = len(kv)
    return kv[key]


def get_data(dirname):
    dirname += "/"
    files = [dirname + _ for _ in os.listdir(dirname) if _.startswith("part")]
    print files
    kv = load_ids_map()
    for f in files:
        print time.ctime(), f
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            items = l.split()
            label = int(items[0])
            feas = map(lambda x: x.split(":"), items[1:])
            fid, value = zip(*feas)
            fid = map(lambda x:kv[x], fid)
            print fid
            value = np.asarray(value, dtype=np.float)
            yield label, fid, value


def load(dirname):
    data = []
    row = []
    col = []
    labels = []
    print "======== load ==========="
    print time.ctime(), "begin"
    for i, (label, fid, fval) in enumerate(get_data(dirname)):
        ct = len(fid)
        row += [[i] * ct]
        col += [fid]
        data += [fval]
        labels += [label]
    print time.ctime(), "end"
    return labels, data, row, col


def trans_to_coo(labels, data, row, col):
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
    return dm


def split(dm):
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


if __name__ == "__main__":
    dirname = sys.argv[1]
    labels, data, row, col = load(dirname)
    dm = trans_to_coo(labels, data, row, col)
    split(dm)
