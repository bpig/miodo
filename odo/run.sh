#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python ferrari.py --top_dir=/home/work/wwxu/dw/$1 --train=20,27 --valid=29,30 --wide_dim=1383386 --deep_dim=288857 --format=new
