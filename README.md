# 使用tensor2tensor机器翻译
自定义训练数据训练transformer
# 用t2t_datagen.py生成训练数据
* **自定义一个用户目录(参数`--t2t_usr_dir`的值)**<br>
该目录下主要存放以下文件：
    1. 自定义问题文件`(myproblem.py)`<br>
    2. 创建`__init__.py`并且在`init.py`中把problem_name 导入，这样才能够被`t2t-datagen`和`t2t-trainer`识别，并注册到t2t里面<br>   
 如下图所示，自定义了用户目录，并在其下自定义问题以及创建__init__.py文件
 ![](https://github.com/orangerfun/tensor2tensor/raw/master/dir.png)




