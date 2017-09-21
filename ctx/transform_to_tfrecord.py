import tensorflow as tf

train_filename = "train.tf"


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_record():
    """return label, fea:value list"""
    filename = "part-00099"
    for l in open(filename):
        l = l.strip()
        if not l:
            continue
        items = l.split("\t")
        assert len(items) == 7
        yield int(items[0]), items[-1].split(' ')


def get_fea_id_map():
    keys = set()
    for _, feas in get_record():
        keys.update(feas)
    keys = sorted(keys)
    return dict([(keys[i], i) for i in range(len(keys))])


def transform_to_tfrecord():
    ids = get_fea_id_map()
    writer = tf.python_io.TFRecordWriter(train_filename)

    for label, feas in get_record():
        fea_id = [ids[_] for _ in feas]
        fea_value = [1] * len(feas)
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': _int64_feature([label]),
                'fea_id': _int64_feature(fea_id),
                'fea_value': _int64_feature(fea_value)
            }
        ))
        writer.write(example.SerializeToString())
    writer.close()

    with open("train.conf", "w") as fout:
        print >> fout, "fea_dim:", len(ids)


if __name__ == "__main__":
    transform_to_tfrecord()
    pass
