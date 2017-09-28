#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python ferrari.py --top_dir=/home/work/wwxu/dw/$1 --train=11,11 --valid=13,13 --wide_dim=25599475 --deep_dim=2361297 --format=new
