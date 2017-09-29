#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python ferrari.py --top_dir=/home/work/wwxu/dw/$1 --train=11,27 --valid=29,30 --wide_dim=25446059 --deep_dim=2361321 --format=new
