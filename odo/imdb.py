# coding:utf-8
# 2017/10/17 下午4:08

# http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl

from data import *
from common import *


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


def infer(fea):
    sparse_dim = 410315
    X = []
    with tf.variable_scope("embed"):
        init = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(sparse_dim)))
        weights = tf.get_variable("weights", [sparse_dim, 128],
                                  initializer=init)
        biases = tf.get_variable("biases", [128], initializer=tf.zeros_initializer)
        for i in range(1, 13):
            x = fea['%df' % i]
            embed = tf.nn.embedding_lookup_sparse(weights, x, None, combiner="sum") + biases
            X += [tf.nn.relu(embed)]

    X = tf.stack(X, axis=1)
    y = tf.to_float(fea['label'])

    with tf.variable_scope("lstm"):
        basic_cell = tf.contrib.rnn.LSTMCell(num_units=128, use_peepholes=True)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
        states = states[-1]

    with tf.variable_scope("dnn"):
        logits = tf.layers.dense(states, 128, activation=tf.nn.relu)
        logits = tf.layers.dense(logits, 1)

    with tf.variable_scope("loss"):
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy)
    return loss


def train():
    fea, fea_valid = read_data()
    loss = infer(fea)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    training_op = optimizer.minimize(loss)
    global_step = tf.train.create_global_step()

    tf.get_variable_scope().reuse_variables()
    loss2 = infer(fea_valid)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    tl = TrainLog()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            while not coord.should_stop():
                gs, _, l, l2 = sess.run([global_step, training_op, loss, loss2])
                tl.run(gs, l, l2)
        except tf.errors.OutOfRangeError as e:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    train()
