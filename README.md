# 1. 环境
自定义训练数据训练transformer,实现中文到英文的翻译<br>
**环境：**<br>
tensorflow 1.14<br>
python 3.6.x<br>
tensor2tensor

# 2.自定义数据训练Tensor2Tensor
## 2.1 自定义一个用户目录(参数`--t2t_usr_dir`的值)

该目录下主要存放以下文件：<br>

    (1). 自定义问题文件 (myproblem.py)
    (2). 创建 __init__.py 并且在 init.py 中把 problem_name 导入，这样才能够被 t2t-datagen 和 t2t-trainer 识别，并注册到 t2t 里面
自定义的问题类名应该是驼峰法命名，定义的问题对应根据驼峰规则用横线隔开，例如定义问题文件是：`translate_enzh_sub32k`，对应类名 `TranslateEnzhSub32k`，如下图所示，自定义了用户目录，并在其下自定义问题以及创建__init__.py文件<br>
               ![](https://github.com/orangerfun/tensor2tensor/raw/master/dir.png)
## 2.2 自定义问题文件
自定义problem是涉及数据处理方面，也就是主要和`t2t_datagen.py`相关，下面先介绍`t2t_datagen.py`的流程，主要流程如下图所示：<br>
![](https://github.com/orangerfun/tensor2tensor/raw/master/image/t2t_datagen.png)
`t2t_datagen.py`中重要的信息就是`generate_data_for_problem[line196]`和`generate_data_for_registered_problem[line198]`,进入后者函数内部发现生成数据的时候主要是`problem.generate_data`函数，也就是Problem 的generate_data中的函数。对应到不同的问题上面有不同的实现，在文本到文本的问题的上面，是像下面这种方式实现的
```python3
class Text2TextProblem(problem.Problem):

.....

  def generate_data(self, data_dir, tmp_dir, task_id=-1):

  ....

    if self.is_generate_per_split:
      for split, paths in split_paths:
        generator_utils.generate_files(self.generate_encoded_samples(data_dir, tmp_dir, split), paths)
    else:
      generator_utils.generate_files(self.generate_encoded_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN), all_paths)

    generator_utils.shuffle_dataset(all_paths, extra_fn=self._pack_fn())
```
从上面的代码可以看出，主要是通过`generate_encoded_samples`生成编码的样本，然后进行生成文件的。所以这里重要的就是 如何生成编码样本了,进入`generate_encoded_samples`函数内部，如下所示：
```python3
def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
   .....
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    return text2text_generate_encoded(generator, encoder, has_inputs=self.has_inputs)
```
就是通过函数`text2text_generate_encoded` 生成编码样本数据,内部`get_or_create_vocab`是编码器，`generate_samples`是样本生成器，`text2text_generate_encoded`细节如下：
```python3
def text2text_generate_encoded(sample_generator, vocab, targets_vocab=None, has_inputs=True):
  targets_vocab = targets_vocab or vocab
  for sample in sample_generator:
    if has_inputs:
      sample["inputs"] = vocab.encode(sample["inputs"])
      sample["inputs"].append(text_encoder.EOS_ID)    # 表示【EOS】,句子结束
    sample["targets"] = targets_vocab.encode(sample["targets"])
    sample["targets"].append(text_encoder.EOS_ID)
    yield sample
```
主要就是通过迭代器，获取样本数据，然后通过编码器进行编码样本数据，然后yield 出去；首先看编码器encoder实现，进入`get_or_create_vocab`函数，如下：
```python3
  def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
    if self.vocab_type == VocabType.CHARACTER:
      encoder = text_encoder.ByteTextEncoder()
    elif self.vocab_type == VocabType.SUBWORD:
      if force_get:
        vocab_filepath = os.path.join(data_dir, self.vocab_filename)
        encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
      else:
        encoder = generator_utils.get_or_generate_vocab_inner(data_dir, self.vocab_filename, self.approx_vocab_size,
            self.generate_text_for_vocab(data_dir, tmp_dir),
            max_subtoken_length=self.max_subtoken_length,
            reserved_tokens=(text_encoder.RESERVED_TOKENS + self.additional_reserved_tokens))
    elif self.vocab_type == VocabType.TOKEN:
      vocab_filename = os.path.join(data_dir, self.vocab_filename)
      encoder = text_encoder.TokenTextEncoder(vocab_filename, replace_oov=self.oov_token)
        ....
    return encoder
```
进入函数后发现主要是根据实际需求选择不同的编码器（**注意：这里的说的编码器是指将文本用数字代替，与后面的模型里面的编码不同**），字符级别的有`text_encoder.SubwordTextEncoder`这也是tensor2tensor中默认的subword方式；词级别的有`text_encoder.TokenTextEncoder`<br>
然后再看如何通过迭代器，获取样本数据，这里`generate_samples`函数细节如下：
```python3
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
   
    raise NotImplementedError()
```
尴尬，没有实现，在此我们参考其他人写的实现方式，如下：
```python3
@registry.register_problem
class TranslateEndeWmtBpe32k(translate.TranslateProblem):
  ... 
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    dataset_path = ("train.tok.clean.bpe.32000"
                    if train else "newstest2013.tok.bpe.32000")
    train_path = _get_wmt_ende_bpe_dataset(tmp_dir, dataset_path)
    token_path = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(token_path):
      token_tmp_path = os.path.join(tmp_dir, self.vocab_filename)
      tf.gfile.Copy(token_tmp_path, token_path)
      with tf.gfile.GFile(token_path, mode="r") as f:
        vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
      with tf.gfile.GFile(token_path, mode="w") as f:
        f.write(vocab_data)
    return text_problems.text2text_txt_iterator(train_path + ".en",  train_path + ".de")
```
主要看最后一句`text_problems.text2text_txt_iterator`，进入查看细节如下：
```python3
def text2text_txt_iterator(source_txt_path, target_txt_path):
  for inputs, targets in zip(
      txt_line_iterator(source_txt_path), txt_line_iterator(target_txt_path)):
    yield {"inputs": inputs, "targets": targets}
```
就是在generate_samples 函数里面提供平行语料的路径，然后使用 text2text_txt_iterator 就能获取对应的样本数据<br>
**总结下,整个流程就是我们把平行语料放入`generate_samples`中通过迭代器取样本出来，然后经过`get_or_create_vocab`函数对取出来的样本进行编码（文本数字化），再将编码后的数据放入`generator_utils.generate_files
`中生成文件，当然中间还有许多细节，现在再去看上面的流程图，感觉一切全部就都连起来了，总的来说我们要改的主要就是`generate_encoded_samples`这个函数**


### 2.2.1 自定义problem--使用tensor2tensor中默认subword工具（使用自己的平行语料）
完整程序见`my_t2t/translate_enzh_fc.py`
### 2.2.22 自定义problem--使用BPE进行subword(使用自己的词表，平行语料)
完整程序见`my_t2t/translate_enzh_bpe.py`
## 2.3 调用t2t_datagen.py生成格式化数据
调用参数设置如下：<br>

    python tensor2tensor/bin/t2t_datagen.py \
    --data_dir=$DATABASE/../t2t_datagen/$CATEGORY \   # 自定义目录，用于存放生成的训练数据
    --tmp_dir=$DATABASE \                             # 平行语料存放的目录
    --problem=translate_enzh_fc                       # 自定义的问题文件名
    --t2t_usr_dir=$CODE_DIR/my_t2t                    # 自定义的用户目录，也就是存放自定义问题的目录
 具体设置参见`train`目录下`zhen_data_gen_fangcheng.sh`<br>
 生成的格式化数据样式如下：<br>
 ![](https://github.com/orangerfun/tensor2tensor/raw/master/image/tfrecord.png)
## 2.4 调用`t2t_trainer.py`训练数据
参数设置及含义如下：<br>

    python tensor2tensor/bin/t2t_trainer.py \
        --data_dir=$DATA_DIR \                 # 存放格式化训练数据的目录
        --t2t_usr_dir=$USER_DIR \              # 存放问题文件的目录
        --problem=$PROBLEM \                   # 问题文件
        --model=$MODEL \                       # 模型
        --hparams_set=$HPARAMS \               # 超参文件
        --output_dir=$TRAIN_DIR \              # 训练文件输出目录
        --worker_gpu=2 \
        --train_steps=200                      # epoch数量
 脚本见`train`目录下`zhen_train_fangcheng.sh`
 ## 2.5 使用`t2t-decoder`预测
 参数设置如下：<br>
 ```
 python tensor2tensor/bin/t2t-decoder \  
    --t2t_usr_dir=$USER_DIR \
    --problem=$PROBLEM \
    --data_dir=$DATA_DIR \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --decode_from_file=$BASEPATH/temp_dir/test.en.seg \      # 用来测的试源文件
    --decode_to_file=$BASEPATH/decode_res/result.txt \       # 测试结果保存的文件
    --checkpoint_path=$TRAIN_DIR/model.ckpt-27000             # 调用已经训练好的模型测试
```
## 2.6 使用t2t-bleu计算测试bleu值
```
 python tensor2tensor/bin/t2t-bleu --translation=$TARGET_FILE --reference=$TRANS_FILE
```
## 2.7 t2t-exporter导出模型
```
cd $CODE_DIR
python ./tensor2tensor/bin/t2t-exporter --t2t_usr_dir=$USR_DIR \
                    --problem=$PROBLEM \
                    --data_dir=$DATA_DIR \
                    --model=$MODEL \
                    --hparams_set=$HPARAMS \
                    --output_dir=$OUTPUT_DIR        #导出模型保存路径
```
## 2.8 部署
**注意：在低版本tensor2tensor中，上面第二个参数名称是problems=$PROBLEM**

 





