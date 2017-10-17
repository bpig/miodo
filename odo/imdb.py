# coding:utf-8
# 2017/10/17 下午4:08

# http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl

from data import *
from common import *


def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)


def max_norm_regularizer(weights, threshold=0.1, axes=1, name="max_norm", collection="max_norm"):
    if threshold == 0.0:
        return
    clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
    clip_weights = tf.assign(weights, clipped, name=name)
    tf.add_to_collection(collection, clip_weights)


class TrainLog():
    def __init__(self, step=10):
        self.aa = 0.0
        self.bb = 0.0
        self.step = step

    def run(self, gs, loss, loss_valid):
        factor = 0.99
        if self.aa == 0.0:
            self.aa = loss
            self.bb = loss_valid
        else:
            self.aa = self.aa * factor + (1 - factor) * loss
            self.bb = self.bb * factor + (1 - factor) * loss_valid
        if gs % self.step == 0:
            print "%s %5d %.3f %.3f %.3f" % (time.ctime(), gs, loss, self.aa, self.bb)


def infer(fea, training=True):
    sparse_dim = 410315
    X = []
    with tf.variable_scope("embed"):
        embed_dim = 48
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(sparse_dim)))
        weights = tf.get_variable("weights", [sparse_dim, embed_dim],
                                  initializer=init)
        biases = tf.get_variable("biases", [embed_dim], initializer=tf.zeros_initializer)
        for i in range(1, 13):
            x = fea['%df' % i]
            embed = tf.nn.embedding_lookup_sparse(weights, x, None, combiner="mean")
            X += [leaky_relu(embed)]

    X1 = tf.stack(X, axis=1)
    X2 = tf.stack(X[::-1], axis=1)
    y = tf.to_float(fea['label'])

    keep_prob = 0.5
    with tf.variable_scope("lstm1"):
        # cell = tf.contrib.rnn.LSTMCell(num_units=8, use_peepholes=True)
        layers = [tf.contrib.rnn.BasicLSTMCell(num_units=12,
                                               activation=tf.nn.relu)
                  for _ in range(2)]
        # cell = tf.contrib.rnn.BasicLSTMCell(num_units=8)
        if training:
            layers = [tf.contrib.rnn.DropoutWrapper(_, input_keep_prob=keep_prob) for _ in layers]
        fw_cell = tf.contrib.rnn.MultiRNNCell(layers)
        # fw_cell = layers[0]
        # outputs, states = tf.nn.dynamic_rnn(cell, X1, dtype=tf.float32)
        # states1 = states[-1][1]
        # states = tf.concat(axis=1, values=states)

    with tf.variable_scope("lstm2"):
        layers = [tf.contrib.rnn.BasicLSTMCell(num_units=12,
                                               activation=tf.nn.relu)
                  for _ in range(2)]
        if training:
            layers = [tf.contrib.rnn.DropoutWrapper(_, input_keep_prob=keep_prob) for _ in layers]
        bw_cell = tf.contrib.rnn.MultiRNNCell(layers)
        # bw_cell = layers[0]
        # outputs, states = tf.nn.dynamic_rnn(cell, X2, dtype=tf.float32)
        # states2 = states[-1][1]

    with tf.variable_scope("bilstm"):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            X1,
            dtype=tf.float32,
        )

        # f, b = states
        # states = tf.concat([f[-1][1], b[-1][1]], 1)

        states = tf.concat(outputs, 2)
        states = tf.reshape(states, (-1, 12))

    with tf.variable_scope("dnn"):
        # states = tf.concat([states1, states2], 1)
        # states = outputs[-1]
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(24.0))
        logits = tf.layers.dense(states, 12, activation=leaky_relu, kernel_initializer=init)
        if training:
            logits = tf.nn.dropout(logits, 0.8)
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(12.0))
        logits = tf.layers.dense(logits, 12, activation=leaky_relu, kernel_initializer=init)
        if training:
            logits = tf.nn.dropout(logits, 0.8)
        logits = tf.layers.dense(logits, 1)

    with tf.variable_scope("loss"):
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy)
    return loss, logits


def dump_pred(ans, fout):
    try:
        prob, label, iid = [_.reshape(-1) for _ in ans]
    except:
        ans[-1] = ans[-1].values
        prob, label, iid = [_.reshape(-1) for _ in ans]

    for v1, v2, v3 in zip(label, prob, iid):
        print >> fout, v1, v2, v3


def restore_model(sess, model_path, use_ema=True):
    global_step = tf.train.get_or_create_global_step()
    if use_ema:
        ema = tf.train.ExponentialMovingAverage(0.991, global_step)
        ema.apply(tf.trainable_variables())
        variables_to_restore = ema.variables_to_restore()

        print variables_to_restore
        saver = tf.train.Saver(variables_to_restore,
                               write_version=tf.train.SaverDef.V2, max_to_keep=10)
    else:
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10)
    saver.restore(sess, model_path)


def pred():
    fea = read_pred()
    _, logits = infer(fea, False)
    prob = tf.sigmoid(logits)
    model_path = "model/imdb-final"

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        tf.local_variables_initializer().run()
        restore_model(sess, model_path, False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        fout = open("lan.ans", "w")

        try:
            while not coord.should_stop():
                ans = sess.run([prob, fea['label'], fea['iid']])
                dump_pred(ans, fout)
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            coord.request_stop()
            coord.join(threads)


def train():
    fea, fea_valid = read_data()
    loss, _ = infer(fea)

    global_step = tf.train.create_global_step()
    lr = tf.train.exponential_decay(
        0.001, global_step, 2000,
        0.1, staircase=True)

    adam = tf.train.AdamOptimizer(learning_rate=lr)
    grads = adam.compute_gradients(loss)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)
    opts = adam.apply_gradients(grads, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(0.991, global_step)

    for var in tf.trainable_variables():
        if not "bias" in var.name:
            max_norm_regularizer(var, 0.95, name=var.name[:-2] + "_norm")

    with tf.control_dependencies([opts]):
        training_op = ema.apply(tf.trainable_variables())

    tf.get_variable_scope().reuse_variables()
    loss2, _ = infer(fea_valid, training=False)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    tl = TrainLog()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10)
        model_path = "model/imdb"

        clip_all_weights = tf.get_collection("max_norm")
        print "max_norm var:"
        print clip_all_weights

        try:
            while not coord.should_stop():
                gs, _, l, l2 = sess.run([global_step, training_op, loss, loss2])
                tl.run(gs, l, l2)
                sess.run(clip_all_weights)
        except tf.errors.OutOfRangeError as e:
            pass
        finally:
            saver.save(sess, model_path + "-final")  # , global_step=global_step)
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        pred()
    else:
        train()
