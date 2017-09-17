# coding:utf-8
# 2017/9/16 下午11:00
# 286287737@qq.com

from common import *


def read():
    data_filename = "part-r-00000"
    filename_queue = tf.train.string_input_producer([data_filename], num_epochs=100)
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)

    batch = tf.train.shuffle_batch(
        [value],
        batch_size=64,
        num_threads=32,
        capacity=50000,
        min_after_dequeue=5000,
        allow_smaller_final_batch=False
    )

    return tf.parse_example(batch, features={
        'label': tf.FixedLenFeature([1], tf.int64),
        'fid': tf.VarLenFeature(tf.int64),
        'fval': tf.VarLenFeature(tf.int64),
    })


def train():
    kv = read()
    sparse_dim = 27088502
    layers = [128, 128, 128]
    fea = tf.sparse_merge(kv['fid'], kv['fval'], sparse_dim)
    glorot = tf.glorot_uniform_initializer
    weights = tf.get_variable("weights", [sparse_dim, layers[0]],
                              initializer=glorot)
    biases = tf.get_variable("biases", [layers[0]], initializer=tf.zeros_initializer)

    embed = tf.nn.embedding_lookup_sparse(
        weights, fea, None, combiner="sum") + biases

    l1 = tf.layers.dense(embed, layers[1], name="l1", activation=tf.nn.relu,
                         kernel_initializer=glorot)
    l2 = tf.layers.dense(l1, layers[2], name="l2", activation=tf.nn.relu,
                         kernel_initializer=glorot)
    logits = tf.layers.dense(l2, 1, name="logists",
                             kernel_initializer=glorot)

    xe = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(kv['label']))
    loss = tf.reduce_mean(xe)

    global_step = tf.contrib.framework.get_or_create_global_step()
    lr = tf.train.exponential_decay(0.001,
                                    global_step,
                                    150,
                                    0.5,
                                    staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            while not coord.should_stop():
                _, loss_value, gs = sess.run([train_op, loss, global_step])
                print gs, loss_value
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            coord.request_stop()
            coord.join(threads)


def stat():
    kv = read()
    mean = tf.reduce_mean(tf.to_float(kv['label']))
    c = 0.0
    ct = 0
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            while not coord.should_stop():
                c += sess.run(mean)
                ct += 1
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            coord.request_stop()
            coord.join(threads)
    print c / ct


if __name__ == "__main__":
    # stat()
    train()
