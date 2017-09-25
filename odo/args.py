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

parser.add_argument('--log_per_batch', type=int, default=100,
                    help='Batch number of log')

parser.add_argument('--save_per_batch', type=int, default=10000,
                    help='Batch number of save model')

parser.add_argument('--valid_per_batch', type=int, default=10000,
                    help='Batch number of do validation')

parser.add_argument('--log_dir', type=str, default='log',
                    help='log directory')

parser.add_argument('--log_name', type=str, default='odo.log',
                    help='log name')

parser.add_argument('--model_dir', type=str, default='model',
                    help='model directory')

parser.add_argument('--model_name', type=str, default='odo',
                    help='model name')

parser.add_argument('--train_dir', type=str, default='',
                    help='training data directory')

parser.add_argument('--valid_dir', type=str, default='',
                    help='validation data directory')

parser.add_argument('--test_dir', type=str, default='',
                    help='test data directory')

parser.add_argument('--predict_out', type=str, default='ans.raw',
                    help='predict output result file')

FLAGS, un_parsed = parser.parse_known_args()


def init_log():
    FORMAT = '%(asctime)-15s\t%(levelname)s\t%(message)s'
    ct = os.listdir(FLAGS.log_dir)
    log_name = FLAGS.log_dir + "/" + FLAGS.log_name + "_%d" % ct
    print log_name
    logging.basicConfig(format=FORMAT, filename=log_name, filemode='w', level=logging.DEBUG)
    logger = logging.getLogger('odo')

    print FLAGS
    logger.info(FLAGS)
    return logger


logger = init_log()
