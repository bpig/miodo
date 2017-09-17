# coding:utf-8
# 2017/9/16 下午11:00
# 286287737@qq.com

from common import *


def read(num_epochs=100):
    data_filename = "part-r-00099"  #160w
    filename_queue = tf.train.string_input_producer(
        [data_filename], num_epochs=num_epochs)
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


def inference(kv):
    # sparse_dim = 27088502
    sparse_dim = 1650679
    layers = [128, 128, 128]
    glorot = tf.uniform_unit_scaling_initializer

    fea = tf.sparse_merge(kv['fid'], kv['fval'], sparse_dim)
    weights = tf.get_variable("weights", [sparse_dim, layers[0]],
                              initializer=glorot)
    biases = tf.get_variable("biases", [layers[0]], initializer=tf.zeros_initializer)

    embed = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="sum") + biases

    l1 = tf.layers.dense(embed, layers[1], name="l1", activation=tf.nn.relu,
                         kernel_initializer=glorot)
    l2 = tf.layers.dense(l1, layers[2], name="l2", activation=tf.nn.relu,
                         kernel_initializer=glorot)
    logits = tf.layers.dense(l2, 1, name="logists",
                             kernel_initializer=glorot)
    return logits


def loss_op(kv, logits):
    xe = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(kv['label']))
    return tf.reduce_mean(xe)


def train_op(loss):
    global_step = tf.train.create_global_step()
    lr = tf.train.exponential_decay(0.001,
                                    global_step,
                                    3000,
                                    0.5,
                                    staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    opt = optimizer.minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(0.99, global_step)
    avg = ema.apply(tf.trainable_variables())
    return tf.group(*[opt, avg])


def train():
    kv = read()
    logits = inference(kv)
    loss = loss_op(kv, logits)
    summary = tf.summary.scalar("loss", loss)
    opt = train_op(loss)
    global_step = tf.train.get_global_step()
    saver = tf.train.Saver()
    model_path = "model/ctx.ckpt"
    log_path = "log_160w"
    writer = tf.summary.FileWriter(logdir=log_path)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            while not coord.should_stop():
                _, loss_value, gs, loss_log = sess.run([opt, loss, global_step, summary])
                print gs, loss_value
                writer.add_summary(loss_log, gs)
                if gs % 10000 == 0:
                    saver.save(sess, model_path, global_step=global_step)
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            saver.save(sess, model_path, global_step=global_step)
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
