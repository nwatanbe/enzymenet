# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

# Tanimoto
def tanimoto(v1, v2):
    ab = np.dot(v1, v2)
    a_norm = np.dot(v1, v1)
    b_norm = np.dot(v2, v2)
    result = ab/(a_norm+b_norm-ab)
    return result


def calc_tanimoto(ref_df, search_df):
    ref_vecs = ref_df.values
    
    tmt_df = []
    for s_na in search_df.index.values.tolist():
        s_vec = search_df.loc[s_na, :].values
        tmt_score = [tanimoto(s_vec, r_vec) for r_vec in ref_vecs ]
        tmt_df.append([s_na] + tmt_score)
    
    tmt_cols = ["name"] + ref_df.index.values.tolist()
    tmt_df = pd.DataFrame(tmt_df, columns=tmt_cols)

    return tmt_df


# 類似度計算の実行
def run_sim(ref_fs, search_fs, v_max, outbase):
    # reference table のロード
    v_cols = [f"v{idx}" for idx in range(1, v_max+1)]
    ref_df = [pd.read_pickle(f) for f in ref_fs]
    ref_df = pd.concat(ref_df, axis=0).set_index("name")
    ref_df = ref_df.loc[:, v_cols]

    # tanimoto 計算
    tmt_df = []
    for search_f in search_fs:
        search_df = pd.read_pickle(search_f).set_index("name")
        search_df = search_df.loc[:, v_cols]
        scr_df = calc_tanimoto(ref_df, search_df)
        tmt_df.append(scr_df)
    
    tmt_df = pd.concat(tmt_df, axis=0).set_index("name")
    scr_cols = tmt_df.columns.values.tolist()
    
    # max 値, mean 値の算出
    tmt_df.loc[:, "max_score"] = tmt_df.max(axis=1)
    tmt_df.loc[:, "mean_score"] = tmt_df.mean(axis=1)

    # 各種結果の保存
    all_f = os.path.join(*[outbase, "all_tanimoto_score.tsv"])
    all_tmt_df = tmt_df.reset_index(drop=False).loc[:, ["name"] + scr_cols + ["max_score", "mean_score"]]
    all_tmt_df.to_csv(all_f, sep="\t", index=False)

    rank_f = os.path.join(*[outbase, "ranking_tanimoto_score.tsv"])
    rank_tmt_df = tmt_df.reset_index(drop=False).loc[:, ["name", "max_score"]].rename(columns={"max_score":"Tanimoto score"})
    rank_tmt_df = rank_tmt_df.sort_values(by=["Tanimoto score"], ascending=False)
    rank_tmt_df.loc[:, "Ranking"] = list(range(1, rank_tmt_df.shape[0]+1))
    rank_tmt_df = rank_tmt_df.loc[:, ["Ranking", "name", "Tanimoto score"]]
    rank_tmt_df.to_csv(rank_f, sep="\t", index=False)