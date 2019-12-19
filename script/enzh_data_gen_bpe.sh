#!/usr/bin/env bash


export PYTHONPATH="/home/rd/translate:$PYTHONPATH"
CODE_DIR="/home/rd/translate"
DATABASE="/home/rd/BPE_seg"

#cd $DATABASE
#tar -cvf train_test.tar.gz train.en.seg train.zh.seg valid.en.seg valid.zh.seg

# t2t data gen
cd $CODE_DIR
echo "start tensor2tensor...\n"
python tensor2tensor/bin/t2t-datagen \
    --data_dir=$DATABASE/t2t_datagen \
    --tmp_dir=$DATABASE/raw_data \
    --problem=translate_enzh_bpe \
    --t2t_usr_dir=$CODE_DIR/my_t2t