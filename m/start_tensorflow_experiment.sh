#!/bin/sh
source  ~/.bashrc

copy_data=1
if [ $# -ge 1 ]; then
    copy_data=$1
fi

TFRecord_path="/user/h_miui_ad/develop/wangpeng9/micontest/TFRecord_base"

cur_path=`pwd`
train_data_path="/home/work/wangpeng9/micontest/data/train"
valid_data_path="/home/work/wangpeng9/micontest/data/valid"

#copy train data
if [ ! -d ${train_data_path} ]; then
    mkdir -p ${train_data_path}
fi
if [ ${copy_data} -eq 1 ]; then
    rm -rf ${train_data_path}/*
    cd ${train_data_path}
    for((day=11;day<=26;day++));
    do
        hadoop --cluster c3prc-hadoop fs -get ${TFRecord_path}/date=${day}
    done
fi

#copy valid data
if [ ! -d ${valid_data_path} ]; then
    mkdir -p ${valid_data_path}
fi
if [ ${copy_data} -eq 1 ]; then
    rm -rf ${valid_data_path}/*
    cd ${valid_data_path}
    for((day=28;day<=30;day++));
    do
        hadoop --cluster c3prc-hadoop fs -get ${TFRecord_path}/date=${day}
    done
fi

#back to work path
cd ${cur_path}

training_dir=${train_data_path}
validation_dir=${valid_data_path}

log_dir=dnn_log
if [ ! -d ${log_dir} ]; then
    mkdir -p ${log_dir}
fi
log_name="dnn_wide_deep.log"

model_dir=dnn_model
if [ ! -d ${model_dir} ]; then
    mkdir -p ${model_dir}
fi
model_name="dnn"

CUDA_VISIBLE_DEVICES="2"  python model/experiment.py \
    --batch_size=128 \
    --is_shuffle=1 \
    --train_dir=${training_dir} \
    --valid_dir=${validation_dir} \
    --model_dir=${model_dir} \
    --model_name=${model_name} \
    --log_dir=${log_dir} \
    --log_name=${log_name} \
    > log.experiment 2>&1
