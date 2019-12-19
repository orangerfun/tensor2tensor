#!/usr/bin/env bash

export PYTHONPATH="/home/rd/translate/:$PYTHONPATH"
CODE_DIR="/home/rd/translate"
DATABASE="/home/rd/BPE_seg"
PROBLEM="translate_enzh_bpe"
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=${DATABASE}/t2t_datagen
TRAIN_DIR=${DATABASE}/train_dir
USER_DIR=${CODE_DIR}/my_t2t

cd $CODE_DIR
python tensor2tensor/bin/t2t-trainer \
    --data_dir=${DATA_DIR} \
    --t2t_usr_dir=${USER_DIR} \
    --problem=${PROBLEM} \
    --model=${MODEL} \
    --hparams_set=${HPARAMS} \
    --output_dir=${TRAIN_DIR} \
    --worker_gpu=4