#!/usr/bin/env bash
BASE_DIR="/home/rd/fangcheng"
export PYTHONPATH="$BASE_DIR/translate:$PYTHONPATH"
CODE_DIR=$BASE_DIR/translate
USR_DIR=$CODE_DIR/my_t2t
DATA_DIR=$BASE_DIR/t2t_datagen

PROBLEM=translate_enzh_fc
MODEL=transformer
HPARAMS=transformer_base
OUTPUT_DIR=$BASE_DIR/train_dir

cd $CODE_DIR
python ./tensor2tensor/bin/t2t-exporter --t2t_usr_dir=$USR_DIR \
                    --problem=$PROBLEM \
                    --data_dir=$DATA_DIR \
                    --model=$MODEL \
                    --hparams_set=$HPARAMS \
                    --output_dir=$OUTPUT_DIR
