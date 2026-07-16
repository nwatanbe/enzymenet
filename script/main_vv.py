# -*- coding: utf-8 -*-
import os
import glob
from re import search
from Bio.SeqIO import parse
from numpy.lib.function_base import select
import pandas as pd
import argparse

from utils import remove_files, clean_len_exaa
from preprocess import run_preprocess
from generate_interlayer_vec import run_generate
from calculate_similarity import run_sim
from umap_plot import run_plot



def get_parser():
    parser = argparse.ArgumentParser(
        prog = "Seq vectorization and visualization",
        description = "This program is to vectorize and visualize sequence.",
        add_help=True
    )

    parser.add_argument("work_dir", help="Work directory", type=str)
    
    # 参照する配列ファイル(形式 fasta)
    parser.add_argument("reference_file", help="Reference file name (format: fasta)", type=str)

    # 探索用配列のファイル(形式 fasta)
    parser.add_argument("search_file", help="search file name (format: fasta)", type=str)

    # 使用するモデルのEC number 1桁目(1, 2, 3, 4, 5, 6, ※EC7は不可)
    parser.add_argument("EC_1d", help="First digit of EC number", type=str)

    # 探索用配列から選抜する配列数 (可視化時に強調表示)
    parser.add_argument("--select_num", help="Number of samples to be selected from search file (highlighted when visualizing).", type=int, default=10)

    # umap　のオプション n_neighbors
    parser.add_argument("--n_neighbors", help="n_neighbors is umap option.", type=int, default=15)

    # umap のオプション metric (euclidean, cosine, correlation から選択)
    parser.add_argument("--metric", help="metric is umap option. you can select euclidean, cosine, or correlation", type=str, default="euclidean")

    args = parser.parse_args()
    
    return args


# ユーザの入力が必要な変数
args = get_parser()
work_dir = args.work_dir
ref_file = args.reference_file
ori_ref_f = os.path.join(*[work_dir, "data", ref_file])

search_file = args.search_file
ori_search_f = os.path.join(*[work_dir, "data", search_file])

ec_1d_list = ["1", "2", "3", "4", "5", "6"]
ec_1d = args.EC_1d
ec_4d = f"EC{ec_1d}_4d"

select_num = args.select_num
n_neighbors = args.n_neighbors
metric = args.metric

# ユーザ入力が不要な変数
v_max = 2048
maxseqlen = 1000
maxlength = 1024
n_jobs = 1
random_state = 12345
pp_batchsize = 50000
gv_batchsize = 128

vocab_f = os.path.join(*["asset", "vocab_no_exAA_no_ClsEos.json"])
outdir = os.path.join(*[work_dir, "result", "vectorize_visualize"])
tf_base = os.path.join(*[outdir, "{}", "batch_v"])
vec_base = os.path.join(*[outdir, "{}", "*.pkl"])
tmt_f = os.path.join(*[outdir, "ranking_tanimoto_score.tsv"])
config_base = os.path.join(*["..", "model", "{}", "{}_model_config.json"])
weight_base = os.path.join(*["..", "model", "{}", "ckpt", "ckpt-{}"])



try:
    # reference ファイルのsequence 確認
    clean_ref_f = ori_ref_f.replace(".fasta", "_clean.fasta")
    flag = clean_len_exaa(ori_ref_f, maxseqlen, clean_ref_f)
    assert flag == True, "All sequences of reference file are length > 1000 or containing BJOUXZ."

    # search ファイルのsequence の確認
    clean_search_f = ori_search_f.replace(".fasta", "_clean.fasta")
    flag = clean_len_exaa(ori_search_f, maxseqlen, clean_search_f)
    assert flag == True, "All sequences of search file are length > 1000 or containing BJOUXZ."

    # EC_1d 入力の確認
    assert ec_1d in ec_1d_list, "EC_1d is not 1-6."


    # reference preprocess
    ref_tf_base = tf_base.format("reference")
    run_preprocess(clean_ref_f, vocab_f, pp_batchsize, maxlength, ref_tf_base)

    # reference generate vec
    ref_tfs = glob.glob(os.path.join(*[outdir, "reference", "*.tfrecord"]))
    run_generate(ec_4d, ref_tfs, config_base, weight_base, gv_batchsize)


    # searchfile preprocess
    search_tf_base = tf_base.format("search")
    run_preprocess(clean_search_f, vocab_f, pp_batchsize, maxlength, search_tf_base)

    # searchfile generate vec
    search_tfs = glob.glob(os.path.join(*[outdir, "search", "*.tfrecord"]))
    run_generate(ec_4d, search_tfs, config_base, weight_base, gv_batchsize)


    # calculate tanimoto
    ref_vec_fs = glob.glob(vec_base.format("reference"))
    search_vec_fs = glob.glob(vec_base.format("search"))
    run_sim(ref_vec_fs, search_vec_fs, v_max, outdir)


    # umap plot
    run_plot(ref_vec_fs, search_vec_fs, tmt_f, v_max, select_num, n_neighbors, n_jobs, metric, random_state, outdir)
    

    # 不要なファイルの削除
    del_files = glob.glob(os.path.join(*[outdir, "*", "*.tfrecord"]))
    remove_files(del_files)



except AssertionError as err:
    print(err)
