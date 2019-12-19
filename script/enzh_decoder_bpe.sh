#!/usr/bin/env bash

export PYTHONPATH="/home/rd/translate/:$PYTHONPATH"
BASEPATH="/home/rd"
CODE_DIR=$BASEPATH/translate
PROBLEM=translate_enzh_bpe
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=$BASEPATH/BPE_seg/t2t_datagen
TRAIN_DIR=$BASEPATH/BPE_seg/train_dir
USER_DIR=$CODE_DIR/my_t2t

cd $CODE_DIR
python tensor2tensor/bin/t2t-decoder \
    --t2t_usr_dir=$USER_DIR \
    --problem=$PROBLEM \
    --data_dir=$DATA_DIR \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --decode_from_file=$BASEPATH/BPE_seg/raw_data/corpus.test.en \
    --decode_to_file=$BASEPATH/BPE_seg/decode_res/result.txt \
    --checkpoint_path=$TRAIN_DIR/model.ckpt-5000