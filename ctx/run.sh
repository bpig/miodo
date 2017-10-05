set -x

gpu=$1
conf=$2.conf

date

CUDA_VISIBLE_DEVICES=$gpu python ferrari.py $conf
    CUDA_VISIBLE_DEVICES=$gpu python ferrari.py $conf pred && \
    ./metrics.py $2

# CUDA_VISIBLE_DEVICES=$2 python ferrari.py $conf pred && ./metrics.py $conf

