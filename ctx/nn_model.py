# coding:utf-8
# 2017/9/16 下午11:00
# 286287737@qq.com

from common import *

cf_str, cf_float, cf_int = read_config(sys.argv[1])


def get_data_list():
    top_dir = cf_str("top_dir")
    ans = []
    for d in range(cf_int("date_begin"), cf_int("date_end") + 1):
        prefix = "%s/date=%2d/" % (top_dir, d)
        ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    return ans


def read():
    data_filename = get_data_list()
    assert len(data_filename)
    print len(data_filename)
    filename_queue = tf.train.string_input_producer(
        data_filename, num_epochs=cf_int("num_epochs"))
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)

    batch = tf.train.shuffle_batch(
        [value],
        batch_size=cf_int("batch_size"),
        num_threads=16,
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
    sparse_dim = cf_int("sparse_dim")
    layer_dim = eval(cf_str("layer_dim"))
    glorot = tf.uniform_unit_scaling_initializer

    fea = tf.sparse_merge(kv['fid'], kv['fval'], sparse_dim)

    with tf.variable_scope("wide"):
        weights = tf.get_variable(
            "weights", [sparse_dim, 1], initializer=tf.zeros_initializer)
        biases = tf.get_variable(
            "biases", [1], initializer=tf.zeros_initializer)
        wide = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="sum") + biases
        wide = tf.sigmoid(wide)

    with tf.variable_scope("embed"):
        weights = tf.get_variable("weights", [sparse_dim, layer_dim[0]],
                                  initializer=glorot)
        biases = tf.get_variable("biases", [layer_dim[0]], initializer=tf.zeros_initializer)

        embed = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="mean") + biases

    with tf.variable_scope("deep"):
        pre_layer = embed
        for i in range(1, len(layer_dim)):
            layer = tf.layers.dense(pre_layer, layer_dim[i], name="layer%d" % i,
                                    activation=tf.nn.relu, kernel_initializer=glorot)
            pre_layer = layer

    with tf.variable_scope("concat"):
        merge_layer = tf.concat([wide, pre_layer], axis=1)
        logits = tf.layers.dense(merge_layer, 1, name="logists",
                                 kernel_initializer=glorot)

    return logits


def loss_op(kv, logits):
    xe = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(kv['label']))
    return tf.reduce_mean(xe)


def train_op(loss):
    global_step = tf.train.create_global_step()
    lr = tf.train.exponential_decay(
        cf_float("lr"), global_step, cf_int("lr_decay_step"),
        cf_float("lr_decay_rate"), staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    opt = optimizer.minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(0.99, global_step)
    avg = ema.apply(tf.trainable_variables())
    return tf.group(*[opt, avg])


def get_model_path():
    model_path = "model/%s/" % sys.argv[1][:-5]
    os.system("mkdir -p %s" % model_path)
    return model_path + "ctx.ckpt"


def get_log_path():
    if not os.path.exists("log"):
        os.mkdir("log")
    log_path = "log/%s_log" % sys.argv[1][:-5]
    fout = open(log_path + "/loss_log", "w")
    return log_path, fout


def train():
    kv = read()
    logits = inference(kv)
    loss = loss_op(kv, logits)

    summary = tf.summary.scalar("loss", loss)
    opt = train_op(loss)

    global_step = tf.train.get_global_step()
    saver = tf.train.Saver()
    model_path = get_model_path()
    log_path, loss_writer = get_log_path()
    writer = tf.summary.FileWriter(logdir=log_path)

    aa = 0.0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            while not coord.should_stop():
                _, loss_value, gs, loss_log = sess.run([opt, loss, global_step, summary])
                factor = 0.99
                if aa == 0.0:
                    aa = loss_value
                else:
                    aa = aa * factor + (1 - factor) * loss_value
                print gs, loss_value, aa
                print >> loss_writer, gs, loss_value, aa
                writer.add_summary(loss_log, gs)
                if gs % cf_int("lr_decay_step") == 0:
                    saver.save(sess, model_path, global_step=global_step)
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            saver.save(sess, model_path, global_step=global_step)
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    train()
