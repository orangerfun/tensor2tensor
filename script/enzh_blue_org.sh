#!/usr/bin/env bash
BASE_DIR="/home/rd"
export PYTHONPATH="$BASE_DIR/:$PYTHONPATH"
TRANS_FILE=$BASE_DIR/decode_res/result.txt
TARGET_FILE=$BASE_DIR/temp_dir/test.zh.seg
CODE_DIR=$BASE_DIR/translate

cd $CODE_DIR
python tensor2tensor/bin/t2t-bleu --translation=$TARGET_FILE --reference=$TRANS_FILE
