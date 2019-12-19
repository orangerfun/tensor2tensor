# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.utils import registry
import tensorflow as tf
from collections import defaultdict


ENZH_BPE_DATASETS = {
    "TRAIN": "corpus.train",
    "DEV": "corpus.valid"
}


def get_enzh_bpe_dataset(directory, filename):
    '''
    :return: 需要训练/验证的平行语料路径
    '''
    train_path = os.path.join(directory, filename)
    if not (tf.gfile.Exists(train_path + ".en") and
            tf.gfile.Exists(train_path + ".zh")):
        raise Exception("there should be some training/dev data in the tmp dir.")
    return train_path


@registry.register_problem
class TranslateEnzhBpe(translate.TranslateProblem):
    """根据英德和英中的问题修改而来，这里是将英德的一个单词表变为中英的两个单词表来进行数据生成。"""

    @property
    def approx_vocab_size(self):
        return 45000

    @property
    # 源语言词表名称
    def source_vocab_name(self):
        return "vocab.bpe.en.%d" % self.approx_vocab_size

    @property
    # 目标语言词表名称
    def target_vocab_name(self):
        return "vocab.bpe.zh.%d" % self.approx_vocab_size

    def get_vocab(self, data_dir, is_target=False):
        """返回的是一个encoder，单词表对应的编码器"""
        vocab_filename = os.path.join(data_dir, self.target_vocab_name if is_target else self.source_vocab_name)
        if not tf.gfile.Exists(vocab_filename):
            raise ValueError("Vocab %s not found" % vocab_filename)
        return text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")


    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        # 读取原始语料存放位置
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset_path = (ENZH_BPE_DATASETS["TRAIN"] if train else ENZH_BPE_DATASETS["DEV"])
        train_path = get_enzh_bpe_dataset(tmp_dir, dataset_path)
        # 读取词表文件
        src_token_path = (os.path.join(data_dir, self.source_vocab_name), self.source_vocab_name)
        tar_token_path = (os.path.join(data_dir, self.target_vocab_name), self.target_vocab_name)
        for token_path, vocab_name in [src_token_path, tar_token_path]:
            if not tf.gfile.Exists(token_path):
                # 如果不在源语言和目标语言的词表，就从temp_dir中拷贝过来（temp_dir是最开始原始语料存放位置，data_dir是t2t_datagen处理后语料存放位置）
                token_tmp_path = os.path.join(tmp_dir, vocab_name)
                tf.gfile.Copy(token_tmp_path, token_path)
                with tf.gfile.GFile(token_path, mode="r") as f:
                    vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
                with tf.gfile.GFile(token_path, mode="w") as f:
                    f.write(vocab_data)
        # 从训练文件中读取训练句子对{inputs：input， targets: target}   这个里面的句子是分词好的句子，中间用空格连接
        return text_problems.text2text_txt_iterator(train_path + ".en",
                                                    train_path + ".zh")


    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        """在生成数据的时候，主要是通过这个方法获取已编码样本的
        args：data_dir:存储t2tdatagen产生的数据
              temp_dir: 原始数据存放地址
        """
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)    # 一个生成器每次产生一个句子对
        encoder = self.get_vocab(data_dir)
        target_encoder = self.get_vocab(data_dir, is_target=True)
        # 将句子编码， has_input=True表示对源语言数据编码，=False表示对目标语言数据编码
        return text_problems.text2text_generate_encoded(generator, encoder, target_encoder,
                                                        has_inputs=self.has_inputs)

    def feature_encoders(self, data_dir):
        source_token = self.get_vocab(data_dir)
        target_token = self.get_vocab(data_dir, is_target=True)
        return {
            "inputs": source_token,
            "targets": target_token,
                }
