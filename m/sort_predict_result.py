import sys


def sort_predict_result():
    wr = open(sys.argv[2], "w")
    instance_id = []
    predict = []
    line_num = 0
    for line in open(sys.argv[1]):
        line_num += 1
        if line_num == 1:
            print >> wr, "%s" % (line.strip())
            continue
        item = line.strip().split(",")
        instance_id.append(int(item[0]))
        predict.append(item[1])
    i_sorted = sorted(xrange(len(instance_id)), key=lambda i: instance_id[i], reverse=False)
    for i in xrange(len(instance_id)):
        print >> wr, "%d,%s" % (instance_id[i_sorted[i]], predict[i_sorted[i]])
    wr.close()


if __name__ == '__main__':
    sort_predict_result()