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
    })


class NFM(object):
    def __init__(self):
        self.random_seed = 1314
        self.sparse_dim = cf_int("sparse_dim")
        self.layer_dim = eval(cf_str("layer_dim"))
        self.batch_norm = False
        self.hidden_factor = cf_int("fm_factor")

    def inference(self, kv):
        tf.set_random_seed(self.random_seed)

        self.fea = tf.sparse_merge(kv['fid'], kv['fval'], self.sparse_dim)

        self._initialize_weights()

        embeds = tf.nn.embedding_lookup(self.weights['emb_weight'], self.fea)
        self.mean_embeds = tf.reduce_sum(embeds, 1)

        self.square_last_embeds = tf.square(self.mean_embeds)

        self.square_embeds = tf.square(embeds)
        self.mean_last_embeds = tf.reduce_sum(self.square_embeds, 1)

        self.FM = 0.5 * tf.subtract(self.square_last_embeds, self.mean_last_embeds)

        for i in range(0, len(self.layer_dim)):
            self.FM = tf.add(tf.matmul(self.FM, self.weights['l%d' % i]), self.weights['b%d' % i])
            self.FM = tf.nn.relu(self.FM)

        self.FM = tf.matmul(self.FM, self.weights['pred']) + self.weights['bias']
        self.emb_bias = tf.nn.embedding_lookup_sparse(self.weights['emb_bias'], self.fea, None, combiner="mean")
        self.out = self.FM + self.emb_bias

    def get_weight_size(self):
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print "#params: %d" % total_parameters

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # tf.layers.batch_normalization()
        bn_train = tf.layers.batch_normalization(x, momentum=0.9, center=True, scale=True, training=True)
        # bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
        #                           is_training=False, reuse=True, trainable=True, scope=scope_bn)
        # z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return tf.no_op()

    def _initialize_weights(self):
        all_weights = dict()
        glorot = tf.uniform_unit_scaling_initializer()
        all_weights['emb_weight'] = tf.get_variable(
            shape=[self.sparse_dim, self.hidden_factor], initializer=glorot, name='feature_embeddings')
        all_weights['emb_bias'] = tf.get_variable(
            shape=[self.sparse_dim, 1], initializer=tf.zeros_initializer, name='feature_bias')

        assert self.layer_dim
        all_weights['l0'] = tf.get_variable(
            name="l0", initializer=glorot, shape=[self.hidden_factor, self.layer_dim[0]], dtype=tf.float32)
        all_weights['b0'] = tf.get_variable(
            name="b0", initializer=tf.zeros_initializer, shape=[self.layer_dim[0]], dtype=tf.float32)
        for i in range(1, len(self.layer_dim)):
            all_weights['l%d' % i] = tf.get_variable(
                name="l%d" % i, initializer=glorot, shape=self.layer_dim[i - 1:i + 1], dtype=tf.float32)
            all_weights['b%d' % i] = tf.get_variable(
                name="b0", initializer=tf.zeros_initializer, shape=[self.layer_dim[0]], dtype=tf.float32)
        all_weights['pred'] = tf.get_variable(
            name="pred", initializer=glorot, shape=[self.layer_dim[-1], 1], dtype=tf.float32)
        all_weights['bias'] = tf.Variable(0.0, name='bias')
        self.weights = all_weights


def inference(kv):
    sparse_dim = cf_int("sparse_dim")
    layer_dim = eval(cf_str("layer_dim"))
    glorot = tf.uniform_unit_scaling_initializer

    fea = tf.sparse_merge(kv['fid'], kv['fval'], sparse_dim)

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
        # merge = tf.concat([wide, pre_layer], 1)
        merge = pre_layer
        logits = tf.layers.dense(merge, 1, name="logists",
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
    model_path = "model/%s/" % sys.argv[1][:-5]
    os.system("mkdir -p %s" % model_path)
    return model_path + "ctx.ckpt"


def get_log_path():
    if not os.path.exists("log"):
        os.mkdir("log")
    log_path = "log/%s_log" % sys.argv[1][:-5]
    os.system("mkdir -p %s" % log_path)
    cmd = "cp ./nn_model.py %s" % log_path
    os.system(cmd)
    fout = open(log_path + "/loss_log", "w")
    return log_path, fout


def train():
    kv = read()
    nfm = NFM()
    # logits = inference(kv)
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
    train()
