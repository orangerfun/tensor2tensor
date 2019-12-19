#!/usr/bin/env bash

echo "start.."

export PYTHONPATH="/home/rd/translate:$PYTHONPATH"
CODE_DIR="/home/rd/translate"
DATABASE="/home/rd/temp_dir"

cd $DATABASE
tar -cvf train_test.tar.gz train.en.seg train.zh.seg valid.en.seg valid.zh.seg

# t2t data gen
cd $CODE_DIR
echo "start tensor2tensor...\n"
python tensor2tensor/bin/t2t-datagen \
    --data_dir=$DATABASE/../t2t_datagen \
    --tmp_dir=$DATABASE \
    --problem=translate_enzh_fc \
    --t2t_usr_dir=$CODE_DIR/my_t2t
