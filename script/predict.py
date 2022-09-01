# -*- coding: utf-8 -*-
"""
tensorflow version 2.1.0
"""
import os
import numpy as np
import time
import json
import glob
import pickle
import pandas as pd

import tensorflow as tf
from modeling import EC_Predictor, transfer_model
from efficient_layers import create_padding_mask

from utils import get_config, get_weight


"""
## check available gpu
physical_devices = tf.config.list_physical_devices('GPU')
assert len(physical_devices) >= 1, "not found available gpu !!"

## gpu memory control
for idx in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[idx], True)
"""


## 読み込んだ tf.Exampleの解析
def _wrapper_parse_function(dims):

    def _parse_function(example):

        ## 特徴量の辞書
        feature_description={
            "seq_token":tf.io.FixedLenFeature([dims[0]], tf.int64),
            "name":tf.io.FixedLenFeature([dims[2]], tf.string),
        }

        parse_items = tf.io.parse_single_example(example, feature_description)

        return parse_items["seq_token"], parse_items["name"]
    
    return _parse_function


# model の build
def build_model(tar_ec, config_dic):
    model_params = config_dic["model_params"]

    # EC 1桁目モデルのbuild
    ec_predictor = EC_Predictor(**model_params)
    
    # EC 4桁目まで予測モデルのbuild
    if tar_ec != "EC_1d":
        ## ダミー input の作成
        dummy_inp = tf.random.uniform((1, 1024), minval=0, maxval=22, dtype=tf.int32)
        dummy_mask = np.ones((1, 1024))
        dummy_mask[0:1, 996:1024] = 0
        dummy_mask = tf.constant(dummy_mask, dtype=tf.float32)
        dummy_mask = dummy_mask[:, :, tf.newaxis]

        ## EC 1桁目モデルのコンパイルと層の分解
        _ = ec_predictor(dummy_inp, False, dummy_mask)
        layer_1st = ec_predictor.layers
        layer_2nd = layer_1st[2].layers[0:-1]

        # 転移学習モデルの重みがロードできるモデルをbuild
        ec_predictor = transfer_model(
            layer_1st[0],
            layer_1st[1],
            layer_2nd,
            config_dic["new_class"],
            config_dic["new_outact"]
        )
    
    return ec_predictor



# predict run
def run_predict(tar_ec, inps, config_base, weight_base, batch_size=128):

    # 各種設定
    config_f = get_config(tar_ec, config_base)
    weight_f = get_weight(tar_ec, weight_base)
    

    with open(config_f, "r") as fo:
        config_dic = json.load(fo)
    
    BATCH_SIZE = batch_size
    PARSE_DIM = config_dic["PARSE_DIM"]

    # tfrecord のparser を作成
    parse_func = _wrapper_parse_function(PARSE_DIM)

    # build_model
    ec_predictor = build_model(tar_ec, config_dic)

    # set optimizer #
    lr = config_dic["lr"]
    optimizer = tf.keras.optimizers.Adam(lr)

    # create ckpt
    ckpt = tf.train.Checkpoint(model=ec_predictor, optimizer=optimizer)

    # load_weight
    ckpt.restore(weight_f)

    # predict
    for inp in inps:
        output = inp.replace(".tfrecord", "_pred.tsv")

        # tf.data の作成
        ds = tf.data.TFRecordDataset(inp).map(parse_func)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(1)

        # predict
        names = []
        preds = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        from_ = 0
        for inp, name in ds:
            batch_num = tf.shape(inp)[0]
            padding_mask = create_padding_mask(inp, 0)
            pred = ec_predictor(inp, False, padding_mask)

            to_ = from_ + batch_num
            preds = preds.scatter(tf.range(from_, to_), pred)
            from_ = to_


            name_ = name.numpy()
            name_ = name_.reshape(name_.shape[0], )
            name_ = list(map(lambda x:x.decode(), list(name_)))
            names.extend(name_)
    
        # データフレームの作成
        preds = preds.stack().numpy()
        assert len(names) == len(preds), "Error !! not same nums"

        df = pd.DataFrame(preds)
        col_s = ["class{}".format(i) for i in range(df.shape[1])]
        df.columns = col_s
        df["name"] = names
        df = df[["name"] + col_s]
        df.to_csv(output, sep="\t", index=False)

        #print("df: {}".format(df.shape), flush=True)


    ## calculation graph clear ##
    tf.keras.backend.clear_session()