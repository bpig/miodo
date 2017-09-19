# coding:utf-8
# 2017/9/16 下午11:00
# 286287737@qq.com

from common import *
from nfm import *
from dnn import *
from data import *


class Env(object):
    def __init__(self, cf, conf_file):
        self.cf = cf
        self.conf_file = conf_file

    def get_model_path(self):
        model_path = "model/%s/" % self.conf_file[:-5]
        os.system("mkdir -p %s" % model_path)

        model_path += "ctx.ckpt"
        if is_pred:
            return model_path + "-" + self.cf.get("pred", "pred_model_step")
        else:
            return model_path

    def get_log_path(self):
        if not os.path.exists("log"):
            os.mkdir("log")
        log_path = "log/%s_log" % self.conf_file[:-5]
        os.system("mkdir -p %s" % log_path)
        cmd = "cp ./ferrari.py %s" % log_path
        os.system(cmd)
        if not is_pred:
            fout = open(log_path + "/loss_log", "w")
        else:
            fout = open(log_path + "/pred_result", "w")
        return log_path, fout


def dump_pred(ans, fout):
    prob, label, iid = [_.reshape(-1) for _ in ans]
    for v1, v2, v3 in zip(label, prob, iid):
        print >> fout, v1, v2, v3


def restore_model(sess, model_path, use_ema=True):
    global_step = tf.train.get_or_create_global_step()
    if use_ema:
        ema = tf.train.ExponentialMovingAverage(0.99, global_step)
        ema.apply(tf.trainable_variables())
        variables_to_restore = ema.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver()
    saver.restore(sess, model_path)


def pred(cf, model, env, data):
    kv = data.read()
    logits = model.inference(kv['fid'])

    prob = tf.sigmoid(logits)

    model_path = env.get_model_path()
    _, fout = env.get_log_path()

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        restore_model(sess, model_path)

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


def train(cf, model, env, data):
    kv = data.read()
    valid_label, valid_fea = data.read_valid()

    fea = tf.placeholder(tf.float32, shape=(None, model.sparse_dim))
    label = tf.placeholder(tf.float32, shape=(None, 1))
    logits = model.inference(fea)
    valid_loss = model.loss_op(label, logits)

    logits = model.inference(kv['fid'])
    loss = model.loss_op(kv['label'], logits)

    summary = tf.summary.scalar("loss", loss)
    opt = model.train_op(loss)

    global_step = tf.train.get_global_step()
    saver = tf.train.Saver()
    model_path = env.get_model_path()
    log_path, loss_writer = env.get_log_path()
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
                if gs % cf.getint("train", "dump_step") * 2 == 0:
                    saver.save(sess, model_path, global_step=global_step)
                print "valid", sess.run(valid_loss, feed_dict={fea: valid_fea, label: valid_label})
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            saver.save(sess, model_path, global_step=global_step)
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    is_pred = True if len(sys.argv) > 2 else False

    conf_file = sys.argv[1]
    cf = ConfigParser()
    cf.read("conf/" + conf_file)
    Model = eval(cf.get("net", "model"))
    model = Model(cf)
    env = Env(cf, conf_file)
    data = Data(cf, is_pred)
    if not is_pred:
        train(cf, model, env, data)
    else:
        pred(cf, model, env, data)
