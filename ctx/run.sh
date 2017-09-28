conf=$1
CUDA_VISIBLE_DEVICES=3 python ferrari.py $conf && CUDA_VISIBLE_DEVICES=3 python ferrari.py $conf pred && ./metrics.py $conf
