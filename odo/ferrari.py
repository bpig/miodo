from common import *
from data import *
from network import *
from args import *

prepare_env()

log_name = FLAGS.dir + "/" + "%d.log" % FLAGS.idx

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.FileHandler(log_name)
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s  %(message)s')

handler.setFormatter(formatter)

logger.addHandler(handler)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.removeHandler(handler)
logger.info(FLAGS)


class TrainLog:
    def __init__(self):
        self.aa = 0.0
        self.bb = 0.0

    def run(self, gs, loss, loss_valid):
        factor = 0.999
        if self.aa == 0.0:
            self.aa = loss
            self.bb = loss_valid
        else:
            self.aa = self.aa * factor + (1 - factor) * loss
            self.bb = self.bb * factor + (1 - factor) * loss_valid
        out = "%4d   %.3f   %.3f   %.3f" % (gs, loss, self.aa, self.bb)
        if gs % FLAGS.log_per_batch == 0:
            print time.ctime(), out
            logger.info(out)


def max_norm(vars, axes=1, name="max_norm", collection="max_norm"):
    if FLAGS.lamda == 0.0:
        return
    logger.info("max_norm")
    threshold = FLAGS.lamda
    for var in vars:
        if "weight" not in var.name:
            continue
        name = var.name[:-2] + "_norm"
        clipped = tf.clip_by_norm(var, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(var, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)


def get_vars():
    vars = tf.trainable_variables()

    wide_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide')
    for var in wide_vars:
        print var.name, var.get_shape()
    print 'wide_vars', len(wide_vars)

    deep_vars = list(set(vars) - set(wide_vars))
    for var in deep_vars:
        print var.name, var.get_shape()
    print 'deep_vars', len(deep_vars)

    return wide_vars, deep_vars


def train():
    fea, valid_fea = read_data()
    layers = eval("[%s]" % FLAGS.layers)

    train_logits, _, _ = inference_deep_wide(fea, layers, FLAGS.keep_prob)

    train_loss = log_loss(train_logits, fea['label'])
    wide_vars, deep_vars = get_vars()
    max_norm(deep_vars)

    global_step = tf.train.get_or_create_global_step()

    ada_optimizer = tf.train.AdagradOptimizer(0.01)

    deep_train_op = ada_optimizer.minimize(
        train_loss, global_step=global_step, var_list=deep_vars)

    ftrl_optimizer = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=1.0)
    wide_train_op = ftrl_optimizer.minimize(
        train_loss, var_list=wide_vars)

    clip_all_weights = tf.get_collection("max_norm")

    ema = tf.train.ExponentialMovingAverage(0.999, global_step)
    with tf.control_dependencies([deep_train_op, wide_train_op]):
        with tf.control_dependencies(clip_all_weights):
            train_op = ema.apply(tf.trainable_variables())

    tf.get_variable_scope().reuse_variables()
    valid_logits, _, _ = inference_deep_wide(valid_fea, layers, 1.0)
    valid_loss = log_loss(valid_logits, valid_fea['label'])

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # path = "model/4/m4-final"
        # restore_model(sess, path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tl = TrainLog()
        try:
            while not coord.should_stop():
                _, t_loss, v_loss, step = sess.run([
                    train_op, train_loss, valid_loss, global_step])

                tl.run(step, t_loss, v_loss)

                if step % FLAGS.save_per_batch == 0:
                    saver.save(sess, FLAGS.model, global_step=step)

                if step >= FLAGS.max_steps:
                    logger.info('break by max_steps: %8d', FLAGS.max_steps)
                    break
        except tf.errors.OutOfRangeError as e:
            pass
        finally:
            saver.save(sess, FLAGS.model + "-final")
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()
