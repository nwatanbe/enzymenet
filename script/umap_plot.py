# -*- coding: utf-8 -*-
import os
import numpy as np
from numba import set_num_threads
import pandas as pd
from scipy.sparse.construct import random
from scipy.sparse.sputils import matrix

import sklearn
from sklearn.preprocessing import StandardScaler

import umap

import matplotlib
matplotlib.use("Agg") #サーバーでプロットするときは必要
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


#seabornのスタイルを設定
sns.set_style("white")
#sns.set_context(context="paper", )

# 論文用のパラメータ設定 #
#plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in
plt.rcParams["font.size"] = 8 # 全体のフォントサイズが変更されます。
plt.rcParams["xtick.bottom"] = True         # 下部に目盛り線を描くかどうか
plt.rcParams["ytick.left"] = True           # 左部に目盛り線を描くかどうか
plt.rcParams["xtick.major.size"] = 3      # x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 3      # y軸主目盛り線の長さ
#plt.rcParams["xtick.major.width"] = 1.0     # x軸主目盛り線の線幅

# 論文用　凡例
plt.rcParams["legend.fancybox"] = False # 丸角
plt.rcParams["legend.framealpha"] = 1.0 # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'lightgray' # edgeの色を変更
plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
#plt.rcParams["legend.labelspacing"] = 5. # 垂直方向（縦）の距離の各凡例の距離
#plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 6 # 点がある場合のmarker scale
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.facecolor"] = "white" # 凡例の背景色


# 次元圧縮
def reduce_dim(matrix, n_neighbors, n_jobs, metric, random_state):
    set_num_threads(n_jobs)
    matrix_norm = StandardScaler().fit_transform(matrix)
    X = umap.UMAP(n_neighbors=n_neighbors, random_state=random_state, n_jobs=n_jobs, min_dist=0.1, spread=1.0, n_epochs=None, metric=metric).fit_transform(matrix_norm)
    return X

# plot
def sample_plot(matrixs, labels, color_ids, output):
    ## figオブジェクトの生成 ##
    fig = plt.figure(figsize=(5, 5), dpi=300)

    ## ax1 オブジェクトの生成 ##
    ax1 = fig.add_subplot(1, 1, 1)  #axesオブジェクト 引数は左から、figの行分割数・figの列分割数・プロット番号

    ## colormap の取得
    #cmap = plt.get_cmap("tab10")
    cmap = sns.color_palette("muted")

    ## scatter plot ##
    for idx in range(len(matrixs)):
        matrix = matrixs[idx]
        ax1.scatter(x=matrix[:, 0], y=matrix[:, 1], s=2.5, color=cmap[color_ids[idx]], alpha=0.8, linewidths=0.5, label=labels[idx])


    ## x, y labelの指定 ##
    ax1.set_xlabel("v1", labelpad=8)
    ax1.set_ylabel("v2", labelpad=8)

    # 凡例を表示
    ax1.legend(bbox_to_anchor=(1.32, 1), loc="upper right").get_frame().set_linewidth(0.5)

    ## plot図の設定と保存 ##
    #plt.subplots_adjust(left=0.1, right=0.8)
    plt.savefig(output, bbox_inches="tight", pad_inches=0.1)
    #plt.savefig(outfile)
    plt.close()


# umap plot の実行
def run_plot(ref_fs, search_fs, tmt_f, v_max, select_num, n_neighbors, n_jobs, metric, random_state, outbase):
    # set vcol
    v_col = [f"v{idx}" for idx in range(1, v_max+1)]

    # reference load
    ref_df = [pd.read_pickle(ref_f) for ref_f in ref_fs]
    ref_df = pd.concat(ref_df, axis=0)
    ref_df.loc[:, "label"] = "References"

    # search load
    search_df = [pd.read_pickle(search_f) for search_f in search_fs]
    search_df = pd.concat(search_df, axis=0)
    search_df.loc[:, "label"] = "Others"

    # reference と pred のマージ
    all_df = pd.concat([ref_df, search_df], axis=0)

    # umapによる次元圧縮
    umap_f = os.path.join(*[outbase, "umap_result.tsv"])
    all_matrix = all_df.loc[:, v_col].values
    umap_x = reduce_dim(all_matrix, n_neighbors, n_jobs, metric, random_state)
    umap_df = pd.DataFrame(umap_x, columns=["v1", "v2"])
    umap_df.loc[:, "name"] = all_df["name"].values.tolist()
    umap_df = umap_df.loc[:, ["name", "v1", "v2"]]
    umap_df.to_csv(umap_f, sep="\t", index=False)

    # tanimoto 結果の load
    tmt_df = pd.read_csv(tmt_f, sep="\t", header=0)
    tmt_df = tmt_df.sort_values(by=["Ranking"], ascending=True)

    # plto にデータを整理
    plot_df = pd.merge(umap_df, all_df.loc[:, ["name", "label"]], on="name", how="inner")
    top_names = tmt_df.iloc[0:select_num, :]["name"].values.tolist()
    plot_df.loc[plot_df["name"].isin(top_names), "label"] = "Select samples"

    labels = ["Others", "Select samples", "References"]
    color_ids = [0, 1, 3]
    matrixs = []
    for lb in labels:
        m_df = plot_df[plot_df["label"]==lb].loc[:, ["v1", "v2"]]
        matrixs.append(m_df.values)
    
    # plot
    outpng = os.path.join(*[outbase, "umap_plot.png"])
    sample_plot(matrixs, labels, color_ids, outpng)