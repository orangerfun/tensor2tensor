#!/usr/bin/env bash

CATEGORY="general"
LAN_DIR="zhen"
export PYTHONPATH="/home/tmxmall/rd/fangcheng/tmxmall-translate:$PYTHONPATH"
CODE_DIR="/home/tmxmall/rd/fangcheng/tmxmall-translate/"   # 代码存放位置
DATABASE="home/tmxmall/rd/fangcheng/nmt.data.segment/"    # 平行语料存放位置

cd $DATABASE
# 生成打包文件
tar -cvf general.tar.gz train.en train.zh test.en test.zh

# t2t data gen，调用t2t_datagen.py生成训练数据
cd $CODE_DIR
python tensor2tensor/bin/t2t_datagen.py \
    --data_dir=$DATABASE/../t2t_datagen/$CATEGORY \   # 自定义目录，用于存放生成的训练数据
    --tmp_dir=$DATABASE \                             # 平行语料存放的目录
    --problem=translate_enzh_fc                       # 自定义的问题文件名
    --t2t_usr_dir=$CODE_DIR/my_t2t                    # 自定义的用户目录，也就是存放自定义问题的目录
