# 使用tensor2tensor机器翻译
自定义训练数据训练transformer
# 用t2t_datagen.py生成训练数据
* **自定义一个用户目录(参数--t2t_usr_dir的值)**<br>
该目录下主要存放以下文件：<br>

    自定义问题文件(myproblem.py)<br>
    __init__.py并且在init.py 中把problem_name 导入，这样才能够被t2t-datagen和t2t-trainer识别，并注册到t2t里面<br>
    
如下图所示，自定义了用户目录，并在其下自定义问题以及创建__init__.py文件




