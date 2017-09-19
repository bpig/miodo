# coding:utf-8
# 2017/9/16 下午11:00
# 286287737@qq.com

from common import *
from nfm import *

conf_file = sys.argv[1]
if len(sys.argv) > 2:
    is_pred = True
else:
    is_pred = False
cf_str, cf_float, cf_int = read_config(conf_file)


def get_data_list():
    top_dir = cf_str("top_dir")
    ans = []
    for d in range(cf_int("date_begin"), cf_int("date_end") + 1):
        prefix = "%s/date=%2d/" % (top_dir, d)
        ans += [prefix + _ for _ in os.listdir(prefix) if _.startswith("part")]
    return ans


def read():
    data_file = cf_str("data_file")
    if not data_file:
        data_file = get_data_list()
    else:
        data_file = eval(data_file)
    assert len(data_file)
    print len(data_file)
    filename_queue = tf.train.string_input_producer(
        data_file, num_epochs=cf_int("num_epochs"))
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
        'iid': tf.FixedLenFeature(1, tf.int64),
    })


def inference(kv):
    sparse_dim = cf_int("sparse_dim")
    layer_dim = eval(cf_str("layer_dim"))
    glorot = tf.uniform_unit_scaling_initializer

    # fea = tf.sparse_merge(kv['fid'], kv['fval'], sparse_dim)

    fea = kv['fid']

    # with tf.variable_scope("wide"):
    #     weights = tf.get_variable(
    #         "weights", [sparse_dim, 1], initializer=glorot)
    #     biases = tf.get_variable(
    #         "biases", [1], initializer=tf.zeros_initializer)
    #     wide = tf.nn.embedding_lookup_sparse(weights, fea, None, combiner="sum") + biases

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
        logits = tf.layers.dense(pre_layer, 1, name="logists",
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

    vars = tf.trainable_variables()
    wide_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide')
    deep_vars = list(set(vars) - set(wide_vars))

    # ftrl = tf.train.FtrlOptimizer(cf_float("ftrl_lr"), l1_regularization_strength=1.0)
    # wide_opt = ftrl.minimize(loss, var_list=wide_vars)

    adam = tf.train.AdamOptimizer(learning_rate=lr)
    deep_opt = adam.minimize(loss, global_step=global_step, var_list=deep_vars)

    ema = tf.train.ExponentialMovingAverage(0.99, global_step)
    avg = ema.apply(tf.trainable_variables())
    return tf.group(*[deep_opt, avg])


def get_model_path():
    model_path = "model/%s/" % conf_file[:-5]
    os.system("mkdir -p %s" % model_path)

    model_path += "ctx.ckpt"
    if is_pred:
        return model_path + "-" + cf_str("pred_model_step")
    else:
        return model_path


def get_log_path():
    if not os.path.exists("log"):
        os.mkdir("log")
    log_path = "log/%s_log" % conf_file[:-5]
    os.system("mkdir -p %s" % log_path)
    cmd = "cp ./nn_model.py %s" % log_path
    os.system(cmd)
    if not is_pred:
        fout = open(log_path + "/loss_log", "w")
    else:
        fout = open(log_path + "/pred_result", "w")
    return log_path, fout


def dump_pred(ans, fout):
    prob, label, iid = [_.reshape(-1) for _ in ans]
    for v1, v2, v3 in zip(prob, label, iid):
        print >> fout, v1, v2, v3


def pred():
    kv = read()
    logits = inference(kv)
    prob = tf.sigmoid(logits)

    saver = tf.train.Saver()
    model_path = get_model_path()
    _, fout = get_log_path()

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        saver.restore(sess, model_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            while not coord.should_stop():
                ans = sess.run([prob, kv['label'], kv['iid']])
                dump_pred(ans, fout)
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            coord.request_stop()
            coord.join(threads)


def train():
    kv = read()
    # logits = inference(kv)
    nfm = NFM(conf_file)
    logits = nfm.inference(kv)
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
    if not is_pred:
        train()
    else:
        pred()
