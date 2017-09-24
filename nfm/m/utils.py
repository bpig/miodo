import tensorflow as tf
import logging
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--hidden_layers', type=str, default="64,64,64",
                    help='Network hidden_layers')

parser.add_argument('--num_embed_features', type=int, default=10000,
                    help='Number of embedding features')
parser.add_argument('--num_deep_features', type=int, default=4000000,
                    help='Number of deep features')
parser.add_argument('--num_wide_features', type=int, default=60000000,
                    help='Number of wide features')

parser.add_argument('--is_shuffle', type=int, default=1,
                    help='is shuffle file list and batch')

parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size to train')
parser.add_argument('--max_steps', type=int, default=200000,
                    help='Max number of steps to train')
parser.add_argument('--num_threads', type=int, default=24,
                    help='Number of threads')

parser.add_argument("--num_gpus", type=int, default=1,
                    help="Total number of gpus for each machine. If you don't use GPU, please set it to '0'")
parser.add_argument('--log_per_batch', type=int, default=100,
                    help='Batch number of log')
parser.add_argument('--save_per_batch', type=int, default=5000,
                    help='Batch number of save model')
parser.add_argument('--valid_per_batch', type=int, default=5000,
                    help='Batch number of do validation')

parser.add_argument('--log_dir', type=str, default='dnn_log',
                    help='log directory')
parser.add_argument('--log_name', type=str, default='dnn_wide_deep.log',
                    help='log name')
parser.add_argument('--model_dir', type=str, default='dnn_model',
                    help='model directory')
parser.add_argument('--model_name', type=str, default='dnn',
                    help='model name')

parser.add_argument('--train_dir', type=str, default='',
                    help='training data directory')
parser.add_argument('--valid_dir', type=str, default='',
                    help='validation data directory')
parser.add_argument('--test_dir', type=str, default='',
                    help='test data directory')

parser.add_argument('--predict_out', type=str, default='predict_result.csv',
                    help='predict output result file')


FLAGS, un_parsed = parser.parse_known_args()
tmp = [sys.argv[0]] + un_parsed

if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)

FORMAT = '%(asctime)-15s\t%(levelname)s\t%(message)s'
log_file_name = FLAGS.log_dir + '/' + FLAGS.log_name
logging.basicConfig(format=FORMAT, filename=log_file_name, filemode='w', level=logging.DEBUG)
logger = logging.getLogger('dnn')

logger.info(tmp)
logger.info(FLAGS)

layers = [int(d) for d in FLAGS.hidden_layers.split(",")]

mem = 0
for d in range(len(layers) - 1):
    mem += (layers[d] * layers[d + 1]) * 4.0 / 1024 / 1024 / 1024  # float32, maybe float16 is enough
logger.info('layers: %s, mem used: %f(Gb)' % (layers, mem))
