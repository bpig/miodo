from common import *

parser = argparse.ArgumentParser()

parser.add_argument('--layers', type=str, default="64,64,64",
                    help='Network hidden_layers')

parser.add_argument('--num_deep_features', type=int, default=3000000,
                    help='Number of deep features')

parser.add_argument('--num_wide_features', type=int, default=6000000,
                    help='Number of wide features')

parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size to train')
parser.add_argument('--max_steps', type=int, default=200000,
                    help='Max number of steps to train')

parser.add_argument('--log_per_batch', type=int, default=10,
                    help='Batch number of log')

parser.add_argument('--save_per_batch', type=int, default=10000,
                    help='Batch number of save model')

parser.add_argument('--valid_per_batch', type=int, default=10000,
                    help='Batch number of do validation')

parser.add_argument('--lamda', type=float, default=0.0)

parser.add_argument('--keep_prob', type=float, default=0.5)

parser.add_argument('--dir', type=str, default='model')

parser.add_argument('--model_name', type=str, default='odo',
                    help='model name')

parser.add_argument('--train_begin', type=int, default=11)
parser.add_argument('--train_end', type=int, default=28)

parser.add_argument('--valid_begin', type=int, default=29)
parser.add_argument('--valid_end', type=int, default=30)

parser.add_argument('--top_dir', type=str, default='')

parser.add_argument('--predict_out', type=str, default='ans.raw',
                    help='predict output result file')

FLAGS, un_parsed = parser.parse_known_args()

def prepare_env():
    top_dir = "model/"
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)

    FLAGS.idx = len(os.listdir(top_dir))
    FLAGS.dir = top_dir + `FLAGS.idx`
    logging.info("dir_path", FLAGS.dir)

    if not os.path.exists(FLAGS.dir):
        os.mkdir(FLAGS.dir)
    FLAGS.model = FLAGS.dir + "/m%d" % FLAGS.idx
    logging.info("model_path", FLAGS.model)

prepare_env()

def init_log():
    log_name = FLAGS.dir + "/" + "%d.log" % FLAGS.idx

    logger = logging.getLogger("odo")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_name)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s  %(message)s')

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.info(FLAGS)

    return logger


logger = init_log()

if __name__ == '__main__':
    
    print FLAGS.idx
