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
# 可以存放多个语料；形式：[["http:..tar.gz(链接)",["train.en", "train.zh"(压缩文件中抽取的文件)]],[语料2，格式同1]，[语料3]]
_NC_TRAIN_DATASETS = [[
    "/home/rd/temp_dir/train_test.tar.gz",
    ["train.en.seg", "train.zh.seg"]
]]

_NC_TEST_DATASETS = [[
    "/home/rd/temp_dir/train_test.tar.gz",
    ("valid.en.seg", "valid.zh.seg")
]]


def create_dummy_tar(tmp_dir, dummy_file_name):
    dummy_file_path = os.path.join(tmp_dir, dummy_file_name)
    if not os.path.exists(dummy_file_path):
        tf.logging.info("Generating dummy file: %s", dummy_file_path)
        tar_dummy = tarfile.open(dummy_file_path, "w:gz")
        tar_dummy.close()
    tf.logging.info("File %s is already exists or created", dummy_file_name)


def get_filename(dataset):
    return dataset[0][0].split("/")[-1]


@registry.register_problem
class TranslateEnzhFc(translate.TranslateProblem):

    # 设定单词表生成大小
    @property
    def vocab_size(self):
        return 45000

    # 超过单词表之后的词的表示，None 表示用元字符替换
    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
        return None

    @property
    def approx_vocab_size(self):
        return 45000

    @property
    def source_vocab_name(self):
        return "vocab.enzh-sub-en.%d" % self.approx_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.enzh-sub-zh.%d" % self.approx_vocab_size

    def get_training_dataset(self, tmp_dir):
        full_dataset = _NC_TRAIN_DATASETS
        # 可以添加一些其他的数据集在这里
        return full_dataset

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        train_dataset = self.get_training_dataset(tmp_dir)
        datasets = train_dataset if train else _NC_TEST_DATASETS
        for item in datasets:
            dummy_file_name = item[0].split("/")[-1]
            create_dummy_tar(tmp_dir, dummy_file_name)
            s_file, t_file = item[1][0], item[1][1]
            if not os.path.exists(os.path.join(tmp_dir, s_file)):
                raise Exception("Be sure file '%s' is exists in tmp dir" % s_file)
            if not os.path.exists(os.path.join(tmp_dir, t_file)):
                raise Exception("Be sure file '%s' is exists in tmp dir" % t_file)

        source_datasets = [[item[0], [item[1][0]]] for item in train_dataset]
        target_datasets = [[item[0], [item[1][1]]] for item in train_dataset]

        # 有词汇表直接编码，没有此汇表会自己创建词表；此处构建的是编码器，同时可以创建词表
        source_vocab = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.source_vocab_name,
            self.approx_vocab_size,
            source_datasets,
            file_byte_budget=1e8)

        target_vocab = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.target_vocab_name,
            self.approx_vocab_size,
            target_datasets,
            file_byte_budget=1e8)

        tag = "train" if train else "dev"
        filename_base = "wmt_enzh_%sk_sub_%s" % (self.approx_vocab_size, tag)
        # 将所有语料连接存入一个文件中
        data_path = translate.compile_data(tmp_dir, datasets, filename_base)
        return text_problems.text2text_generate_encoded(
                                        text_problems.text2text_txt_iterator(data_path + ".lang1", data_path + ".lang2"),
                                        source_vocab,
                                        target_vocab)

    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
        target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
        return {
            "inputs": source_token,
            "targets": target_token,
        }
