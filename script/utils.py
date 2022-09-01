# -*- coding: utf-8 -*-
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# clean seq length, exAA
def clean_len_exaa(inp, maxlength, output):
    pattern = r"[BJOUXZ]"
    
    records = []
    with open(inp, "r") as fo:
        for rec in SeqIO.parse(fo, "fasta"):
            seqstr = str(rec.seq)
            if len(seqstr) > maxlength or re.search(seqstr, pattern):
                continue

            records.append(rec)
    
    if len(records) > 0:
        with open(output, "w") as go:
            SeqIO.write(records, go, "fasta")
        return True
    
    else:
        return False
    

# create fasta dic
def create_dict(inp, output):
    with open(inp, "r") as fo:
        fasta_dic = {str(rec.description):str(rec.seq) for rec in SeqIO.parse(fo, "fasta")}
    
    with open(output, "wb") as go:
        pickle.dump(fasta_dic, go)

def get_weight(tar_ec, basepath):
    #basepath = os.path.join(*["..", "model", "{}" "ckpt", "ckpt-{}"])
    weight_f = None

    if tar_ec == "EC_1d":
        weight_f = basepath.format(tar_ec, "200")

    elif tar_ec == "EC1_4d":
        weight_f = basepath.format(tar_ec, "400")
    
    elif tar_ec == "EC2_4d":
        weight_f = basepath.format(tar_ec, "500")
    
    elif tar_ec == "EC3_4d":
        weight_f = basepath.format(tar_ec, "400")
    
    elif tar_ec == "EC4_4d":
        weight_f = basepath.format(tar_ec, "400")
    
    elif tar_ec == "EC5_4d":
        weight_f = basepath.format(tar_ec, "90")
    
    elif tar_ec == "EC6_4d":
        weight_f = basepath.format(tar_ec, "300")

    else:
        weight_f = None
    
    return weight_f


def get_config(tar_ec, basepath):
    #basepath = os.path.join(*["..", "model", "{}", "{}_model_config.json"])
    config_f = None

    if tar_ec == "EC_1d":
        config_f = basepath.format(tar_ec, tar_ec)

    elif tar_ec == "EC1_4d":
        config_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC2_4d":
        config_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC3_4d":
        config_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC4_4d":
        config_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC5_4d":
        config_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC6_4d":
        config_f = basepath.format(tar_ec, tar_ec)

    else:
        config_f = None
    
    return config_f



def get_label(tar_ec, basepath):
    #basepath = os.path.join(*["..", "model", "{}", "{}_label_pair.tsv"])
    label_f = None

    if tar_ec == "EC_1d":
        label_f = basepath.format(tar_ec, tar_ec)

    elif tar_ec == "EC1_4d":
        label_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC2_4d":
        label_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC3_4d":
        label_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC4_4d":
        label_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC5_4d":
        label_f = basepath.format(tar_ec, tar_ec)
    
    elif tar_ec == "EC6_4d":
        label_f = basepath.format(tar_ec, tar_ec)

    else:
        label_f = None
    
    return label_f


def arrange_predict(tar_ec, basedir, lb_path):
    # 各種ファイルパスの設定
    #basedir = os.path.join(*["..", "result"])
    pred_fs = glob.glob(os.path.join(*[basedir, tar_ec, "batch_*pred.tsv"]))
    lb_f = get_label(tar_ec, lb_path)
    output = os.path.join(*[basedir, tar_ec, f"{tar_ec}_arrange_pred.tsv"])

    # lb_f の読込と辞書化
    lb_df = pd.read_csv(lb_f, sep="\t", header=0, dtype="str")
    lb_dic = dict(lb_df.loc[:, ["label", "ec_group"]].values.tolist())

    # score_table の作成
    lb_col = None
    scr_col = None
    if tar_ec == "EC_1d":
        lb_col = "EC_1d_label"
        scr_col = "EC_1d_score"
    else:
        lb_col = "EC_4d_label"
        scr_col = "EC_4d_score"
    

    pred_arg_df = []
    for pred_f in pred_fs:
        pred_df = pd.read_csv(pred_f, sep="\t", header=0).set_index("name").rename(columns=lb_dic)
        pred_lb_df = pd.DataFrame(pred_df.idxmax(axis=1), columns=[lb_col])
        pred_scr_df = pd.DataFrame(pred_df.max(axis=1), columns=[scr_col])
        pred_mrg_df = pd.merge(pred_lb_df, pred_scr_df, left_index=True, right_index=True, how="inner")
        pred_arg_df.append(pred_mrg_df)
    
    pred_arg_df = pd.concat(pred_arg_df, axis=0).reset_index(drop=False).loc[:, ["name", lb_col, scr_col]]
    pred_arg_df.to_csv(output, sep="\t", index=False)


def create_bio_record(na, seqstr):
    seq_obj = Seq(seqstr)
    id_, doc = na.split(" ", 1)
    record = SeqRecord(seq_obj, id=id_, description=doc)
    return record



def separate_fasta(fasta_dict_f, pred_arg_f, outbase):
    # set file path and params
    #pred_arg_f = os.path.join(*["..", "result", "EC_1d", "EC_1d_arrange_pred.tsv"])
    #outbase = os.path.join(*["..", "result", "EC{}_4d", "EC_1d_select.fasta"])

    # file load
    lb_col = "EC_1d_label"
    pred_arg_df = pd.read_csv(pred_arg_f, sep="\t", header=0, dtype={lb_col:"str"})
    
    with open(fasta_dict_f, "rb") as fo:
        fasta_dic = pickle.load(fo)
    
    # separate fasta
    for ec, gdf in pred_arg_df.groupby(lb_col):
        if ec == "Not Enzyme":
            continue

        output = outbase.format(ec)
        records = [create_bio_record(na, fasta_dic[na]) for na in gdf["name"].values.tolist()]
        with open(output, "w") as go:
            SeqIO.write(records, go, "fasta")


def merge_pred_table(pred_1d_f, pred_4d_fs, output):
    
    # pred_1d_df load
    lb_1d_col = "EC_1d_label"
    scr_1d_col = "EC_1d_score"
    pred_1d_df = pd.read_csv(pred_1d_f, sep="\t", header=0, dtype={lb_1d_col:"str"})

    # pred_4d_dfs load
    lb_4d_col = "EC_4d_label"
    scr_4d_col = "EC_4d_score"
    
    if len(pred_4d_fs) == 0:
        ddf = pred_1d_df.copy()
        ddf.loc[:, lb_4d_col] = np.nan
        ddf.loc[:, scr_4d_col] = np.nan
        ddf.to_csv(output, sep="\t", index=False)
    
    else:
        pred_4d_dfs = []
        for inp in pred_4d_fs:
            sub_4d_df = pd.read_csv(inp, sep="\t", header=0, dtype={lb_4d_col:"str"})
            pred_4d_dfs.append(sub_4d_df)
    
        pred_4d_df = pd.concat(pred_4d_dfs, axis=0)

        # merge table
        ddf = pd.merge(pred_1d_df, pred_4d_df, on="name", how="outer")
        ddf = ddf.loc[:, ["name", lb_1d_col, scr_1d_col, lb_4d_col, scr_4d_col]]
        ddf.to_csv(output, sep="\t", index=False)


# 複数ファイルの削除
def remove_files(file_paths):
    for f in file_paths:
        if os.path.exists(f):
            os.remove(f)