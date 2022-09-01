# -*- coding: utf-8 -*-
import os
import glob
from Bio.SeqIO import parse
import pandas as pd
import argparse

from utils import create_dict, arrange_predict, separate_fasta, merge_pred_table, remove_files, clean_len_exaa
from preprocess import run_preprocess
from predict import run_predict



def get_parser():
    parser = argparse.ArgumentParser(
        prog = "EC_Predictor",
        description = "This program is to predict EC number",
        add_help=True
    )

    parser.add_argument("fasta_file", help="Fasta file name", type=str)
    
    args = parser.parse_args()
    
    return args


# ユーザの入力が必要な変数
args = get_parser()
fasta_name = args.fasta_file
origin_fasta_f = os.path.join(*["..", "data", fasta_name])

# ユーザ入力が不要な変数
tar_ec1d = "EC_1d"
tar_ec4d_b = "EC{}_4d"
ec_nums = ["1", "2", "3", "4", "5", "6"]
pp_batchsize = 50000
maxlength = 1024
maxseqlen = 1000
pred_batchsize = 128

fasta_dic_f = os.path.join(*["..", "data", "name_seq_dic.pkl"])
vocab_f = os.path.join(*["asset", "vocab_no_exAA_no_ClsEos.json"])

outdir = os.path.join(*["..", "result", "ec_number"])
tfrecord_base = os.path.join(*[outdir, "{}", "batch_v"])
config_base = os.path.join(*["..", "model", "{}", "{}_model_config.json"])
weight_base = os.path.join(*["..", "model", "{}", "ckpt", "ckpt-{}"])
lb_base = os.path.join(*["..", "model", "{}", "{}_label_pair.tsv"])
pred_arg_base = os.path.join(*[outdir, "{}", "{}_arrange_pred.tsv"])
separate_base = os.path.join(*[outdir, "EC{}_4d", "EC_1d_select.fasta"])
final_output = os.path.join(*[outdir, "EC_predict_final_result.tsv"])


try:
    # maxlength を超える　または　例外文字を含む配列は削除　もし削除後に配列が 0 になった場合はerror として処理を終了する
    clean_fasta_f = origin_fasta_f.replace(".fasta", "_clean.fasta")
    flag = clean_len_exaa(origin_fasta_f, maxseqlen, clean_fasta_f)
    assert flag == True, "All sequences are length > 1000 or containing BJOUXZ."

    # name, sequence の辞書を作成
    create_dict(clean_fasta_f, fasta_dic_f)

    # preprocess によるtfrecor の作成
    ec1d_tfrecord_base = tfrecord_base.format(tar_ec1d)
    run_preprocess(clean_fasta_f, vocab_f, pp_batchsize, maxlength, ec1d_tfrecord_base)

    # predict
    ec1d_tfs = glob.glob(os.path.join(*[outdir, tar_ec1d, "*.tfrecord"]))
    run_predict(tar_ec1d, ec1d_tfs, config_base, weight_base, pred_batchsize)

    # 予測結果の整理
    arrange_predict(tar_ec1d, outdir, lb_base)

    # EC_1d 結果をもとにfasta ファイルのseparate
    ec1d_pred_f = pred_arg_base.format(tar_ec1d, tar_ec1d)
    separate_fasta(fasta_dic_f, ec1d_pred_f, separate_base)

    # EC_4d  4桁目までの予測
    for ec in ec_nums:
        ec4d_fasta_f = glob.glob(separate_base.format(ec))
        if len(ec4d_fasta_f) == 0:
            continue
        else:
            ec4d_fasta_f = ec4d_fasta_f[0]

        tar_ec4d = tar_ec4d_b.format(ec)
        # preprocess によるtfrecor の作成
        ec4d_tfrecord_base = tfrecord_base.format(tar_ec4d)
        run_preprocess(ec4d_fasta_f, vocab_f, pp_batchsize, maxlength, ec4d_tfrecord_base)

        # predict
        ec4d_tfs = glob.glob(os.path.join(*[outdir, tar_ec4d, "*.tfrecord"]))
        run_predict(tar_ec4d, ec4d_tfs, config_base, weight_base)

        # 予測結果の整理
        arrange_predict(tar_ec4d, outdir, lb_base)

    # EC_1d, EC_4d 結果のマージ
    pred_1d_f = pred_arg_base.format(tar_ec1d, tar_ec1d)
    pred_4d_fs = glob.glob(os.path.join(*[outdir, "EC*_4d", "*_arrange_pred.tsv"]))
    merge_pred_table(pred_1d_f, pred_4d_fs, final_output)

    # 不要なファイルの削除
    del_fas = glob.glob(os.path.join(*[outdir, "*", "*.fasta"]))
    del_tfds = glob.glob(os.path.join(*[outdir, "*", "*.tfrecord"]))
    remove_files(del_fas) 
    remove_files(del_tfds)

except AssertionError as err:
    print(err)
