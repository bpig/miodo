from common import *
from inputs import *
from network import *


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
        out = "%d %.3f %.3f %.3f" % (gs, loss, self.aa, self.bb)
        print out


def train():
    with tf.device('/cpu:0'):

        training_fq = tf.train.string_input_producer(all_train_files)
        training_batch = read_batch(training_fq, FLAGS.is_shuffle)

        validation_fq = tf.train.string_input_producer(all_valid_files, num_epochs=1)
        validation_batch = read_all_batch(validation_fq)

    # build training net
    training_logits, _, _ = inference_deep_wide(training_batch['deep_feature_index'],
                                                training_batch['deep_feature_id'],
                                                training_batch['wide_feature_index'],
                                                training_batch['wide_feature_id'],
                                                training_batch['instance_id'],
                                                layers, 0.5)

    # build loss function
    training_sum_loss, training_mean_loss = log_loss(training_logits,
                                                     training_batch['label'])

    trainable_var = tf.trainable_variables()

    trainable_wide_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_hidden1')
    for var in trainable_wide_var:
        print(var, var.name, var.get_shape())
    print 'trainable_wide_var', len(trainable_wide_var)

    trainable_deep_var = []
    for var in trainable_var:
        if var not in trainable_wide_var:
            trainable_deep_var.append(var)
            print(var, var.name, var.get_shape())
    print 'trainable_deep_var', len(trainable_deep_var)

    def max_norm_regularizer(weights, axes=1, name="max_norm", collection="max_norm"):
        threshold = 0.1
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)

    for var in trainable_deep_var:
        if "weight" in var.name:
            max_norm_regularizer(var, name=var.name[:-2] + "_norm")

    # global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # deep training optimizer, update the net parameters
    ada_optimizer = tf.train.AdagradOptimizer(0.01)

    deep_train_op = ada_optimizer.minimize(training_mean_loss,
                                           global_step=global_step,
                                           var_list=trainable_deep_var)

    # wide training optimizer, update the net parameters
    ftrl_optimizer = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=1.0)
    wide_train_op = ftrl_optimizer.minimize(training_mean_loss,
                                            var_list=trainable_wide_var)

    clip_all_weights = tf.get_collection("max_norm")

    ema = tf.train.ExponentialMovingAverage(0.999, global_step)
    with tf.control_dependencies([deep_train_op, wide_train_op]):
        with tf.control_dependencies([clip_all_weights]):
            train_op = ema.apply(tf.trainable_variables())

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # model saved path
        model_out = "model/zl"
        print "model_out =", model_out
        total_training = 0
        ll = TrainLog()
        try:
            while not coord.should_stop():
                start = time.time()

                _, sum_loss, mean_loss, step = sess.run([train_op,
                                                         training_sum_loss, training_mean_loss, global_step])

                end = time.time()
                total_training = end - start

                if step % FLAGS.log_per_batch == 0:
                    ll.run(step, sum_loss, mean_loss)

                # save model
                if step % FLAGS.save_per_batch == 0:
                    saver.save(sess, model_out, global_step=step)

                # validation
                if step % FLAGS.valid_per_batch == 0:
                    start = time.time()

                    _, validation_mean_loss, validation_auc = calc_deep_wide_metrics(sess, validation_batch)

                    end = time.time()
                    total_validation = end - start
                    l = 'step: %09d, training_time: %f, validation_time: %f, validation_mean_loss: %f, validation_auc: %f' % (
                        step, total_training, total_validation, validation_mean_loss, validation_auc)
                    print l
                    logger.info(l)

                if step >= FLAGS.max_steps:
                    logger.info('training finished with step: %09d, max_steps: %09d', step, FLAGS.max_steps)
                    break
        except tf.errors.OutOfRangeError as e:
            pass
        finally:
            saver.save(sess, model_out + "-final")
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()
