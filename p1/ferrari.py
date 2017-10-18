# coding:utf-8

from common import *
from nfm import *
from dnn import *
from dnn_lan import *
from dnn_fc import *
from wde import *
from ftrl import *
from afm import *
from multi_dnn import *
from data import *


class Env(object):
    def __init__(self, conf_file):
        self.conf_file = conf_file

    def get_model_path(self):
        model_path = "model/%s/" % self.conf_file[:-5]
        os.system("mkdir -p %s" % model_path)

        model_path += "ctx.ckpt"
        if is_pred:
            if len(sys.argv) == 4:
                idx = sys.argv[3]
            else:
                idx = "final"
            return model_path + "-" + idx
        else:
            return model_path

    def get_log_path(self):
        if not os.path.exists("log"):
            os.mkdir("log")
        log_path = "log/%s_log" % self.conf_file[:-5]
        os.system("mkdir -p %s" % log_path)
        os.system("cp ./nfm.py %s" % log_path)
        os.system("cp ./dnn.py %s" % log_path)
        if not is_pred:
            fout = open(log_path + "/loss_log", "w")
        else:
            fout = open(log_path + "/pred_result", "w")
        return log_path, fout


def dump_pred(ans, fout):
    try:
        prob, label, iid = [_.reshape(-1) for _ in ans]
    except:
        ans[-1] = ans[-1].values
        prob, label, iid = [_.reshape(-1) for _ in ans]

    for v1, v2, v3 in zip(label, prob, iid):
        print >> fout, v1, v2, v3


def restore_model(sess, model_path):
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10)
    saver.restore(sess, model_path)


def pred(cf, model, env, data):
    kv = data.read(model.feature_map)
    if model.model == "WDE":
        logits, _, _ = model.inference(kv, 0.0)
    else:
        logits = model.inference(kv, 0.0)

    prob = tf.sigmoid(logits)

    model_path = env.get_model_path()
    _, fout = env.get_log_path()

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        tf.local_variables_initializer().run()
        restore_model(sess, model_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        ct = 0
        try:
            while not coord.should_stop():
                ans = sess.run([prob, kv['label'], kv['iid']])
                dump_pred(ans, fout)
                ct += 1
                if ct % 100 == 0:
                    print time.ctime(), ct
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            coord.request_stop()
            coord.join(threads)


def train(cf, model, env, data):
    try:
        drop = cf.getfloat("net", "drop")
    except:
        drop = 0.5
    kv = data.read(model.feature_map)
    kv_valid = data.read_valid(model.feature_map)

    if model.model == "WDE":
        logits, w, d = model.inference(kv, drop)
    else:
        logits = model.inference(kv, drop)
    loss = model.loss_op(kv['label'], logits)

    opt = model.train_op(loss)

    tf.get_variable_scope().reuse_variables()
    if model.model == "WDE":
        logits, _, _ = model.inference(kv_valid, 0.0)
    else:
        logits = model.inference(kv_valid, 0.0)
    loss2 = model.loss_op(kv_valid['label'], logits)

    global_step = tf.train.get_global_step()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10)
    model_path = env.get_model_path()
    log_path, loss_writer = env.get_log_path()

    log = TrainLog(loss_writer, model.step)

    gpu_options = tf.GPUOptions(allow_growth=True)

    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        restore_model(sess, model_path + "-final")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            while not coord.should_stop():
                loss_value, _, gs, loss_valid = sess.run(
                    [loss, opt, global_step, loss2], feed_dict={model.training: True})
                log.run(gs, loss_value, loss_valid)
                if gs % cf.getint("train", "dump_step") == 0:
                    saver.save(sess, model_path, global_step=global_step)
        except tf.errors.OutOfRangeError:
            print "up to epoch limits"
        finally:
            saver.save(sess, model_path + "-final")
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    tf.set_random_seed(131314)
    np.random.seed(131314)

    is_pred = True if len(sys.argv) > 2 else False

    conf_file = sys.argv[1]
    cf = ConfigParser()
    cf.read("conf/" + conf_file)
    Model = eval(cf.get("net", "model"))
    model = Model(cf)
    env = Env(conf_file)
    data = Data(cf, is_pred)
    if not is_pred:
        train(cf, model, env, data)
    else:
        pred(cf, model, env, data)
