# -*- coding: utf-8 -*-
"""
tensorflow 2.1.0
tfrecordへの書き出しとロードについて簡単にまとめたファイル
"""
import tensorflow as tf

import os
import re
import glob
import json
from collections import OrderedDict

import numpy as np
from Bio import SeqIO

import sequence_tokenizer as seq_tk


## tf.Example と互換のある tf.train.Feature に変換する関数(ショートカット)
def _bytes_feature(value):
    # string / byte -> byte_list
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() #BytesListはEagerTensorに対応していないので、numpyに変換する
    elif isinstance(value, list):
        value = value
    else:
        value = [value]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(value)))

def _float_feature(value):
    # float / double -> float_list
    if isinstance(value, list):
        value = value
    elif isinstance(value, type(np.array([1]))):
        value = list(value)
    else:
        value = [value]

    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))

def _int64_feature(value):
    # bool / enum / int / uint -> Int64_list を返す
    if isinstance(value, list):
        value = value
    elif isinstance(value, type(np.array([1]))):
        value = list(value)
    else:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



## tf.Exampleの作成
def create_example(seq_tokens, name):

    # 特徴量名とtf.train.Feature型の対応辞書を作成
    features = OrderedDict()
    features["seq_token"] = _int64_feature(seq_tokens)
    features["name"] = _bytes_feature(name)

    # tf.train.Example による作成
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example.SerializeToString()  #バイナリ型に変換



def generate_batch(fasta_f, batch_size, conv_name=False):
    with open(fasta_f, "r") as fo:
        res = []
        cnt = 0
        for i in SeqIO.parse(fo, "fasta"):
            cnt += 1
            seq_str = str(i.seq)
            ori_name = str(i.description)

            if conv_name:
                new_name = "v{}".format(cnt)
                res.append((ori_name, new_name, seq_str))
            
            else:
                res.append((ori_name, seq_str))


            if len(res) == batch_size:
                yield res
                res = []
        
        if len(res) > 0:
            yield res
            res = []




def preprocess(itms, out_f, dic_f, max_length):
    
    # aa_dic のセットアップ
    with open(dic_f, "r") as fi:
        aa_dic = json.load(fi)
    
    pad_id = aa_dic.get("<pad>")
    cls_id = aa_dic.get("<cls>")
    eos_id = aa_dic.get("<eos>")
    unk_id = aa_dic.get("<unk>")


    with tf.io.TFRecordWriter(out_f) as writer:
        for seq_name, seq_str in itms:

            ## string => byte
            seq_name_b = seq_name.encode()

            ## tokenize ##
            seq_tokens = seq_tk.convert_seq_to_token(
                seq_str,
                aa_dic,
                pad_id,
                unk_id,
                max_length,
                cls_id,
                eos_id
                )
            seq_tokens = list(seq_tokens)

            ## tf example 化 ##
            example = create_example(seq_tokens, seq_name_b)

            ## tfrecordへの書き込み ##
            writer.write(example)



def run_preprocess(inp, dic_f, batch_size, maxlength, outbase):
    # create generator
    g_batch = generate_batch(inp, batch_size, False)

    # tfrecord の作成
    cnt = 0
    for batch_ in g_batch:
        cnt += 1
        output = outbase + f"{cnt}.tfrecord"
        preprocess(batch_, output, dic_f, maxlength)
