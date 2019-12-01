# 使用tensor2tensor机器翻译
自定义训练数据训练transformer
# 用t2t_datagen.py生成训练数据
### 自定义一个用户目录(参数`--t2t_usr_dir`的值)

该目录下主要存放以下文件：<br>

    (1). 自定义问题文件 (myproblem.py)
    (2). 创建 __init__.py 并且在 init.py 中把 problem_name 导入，这样才能够被 t2t-datagen 和 t2t-trainer 识别，并注册到 t2t 里面
        
如下图所示，自定义了用户目录，并在其下自定义问题以及创建__init__.py文件<br>
               ![](https://github.com/orangerfun/tensor2tensor/raw/master/dir.png)
### 自定义问题文件
自定义的问题文件格式参考mt_t2t目录下`translate_enzh_fc.py`<br>
**注意事项：** 自定义的类名应该是驼峰法命名，定义的问题对应根据驼峰规则用横线隔开，例如定义问题文件是：`translate_enzh_sub32k`，对应类名 `TranslateEnzhSub32k`
### 调用t2t_datagen.py生成格式化数据
调用参数设置如下：<br>

    python tensor2tensor/bin/t2t_datagen.py \
    --data_dir=$DATABASE/../t2t_datagen/$CATEGORY \   # 自定义目录，用于存放生成的训练数据
    --tmp_dir=$DATABASE \                             # 平行语料存放的目录
    --problem=translate_enzh_fc                       # 自定义的问题文件名
    --t2t_usr_dir=$CODE_DIR/my_t2t                    # 自定义的用户目录，也就是存放自定义问题的目录
 具体设置参见`train`目录下`zhen_data_gen_fangcheng.sh`<br>
 生成的格式化数据样式如下：<br>
 ![](https://github.com/orangerfun/tensor2tensor/raw/master/tfrecord.png)





