# -*- coding: utf-8 -*-
import json
import numpy as np


def convert_seq_to_token(seq_str, aa_dic, pad_id, unk_id, max_length, cls_id=None, eos_id=None):
    ids = np.full((max_length, ), pad_id)
    
    seq_ids = []
    for aa in seq_str.upper():
        res = aa_dic.get(aa)
        if res == None:
            seq_ids.append(unk_id)
        else:
            seq_ids.append(res)

    if cls_id != None and eos_id != None:
        seq_ids = [cls_id] + seq_ids + [eos_id]
    
    ids[0:len(seq_ids)] = seq_ids
    ids = ids.astype(np.int32)
    return ids




def create_aadic(ex_aa=False, cls_eos=False):
    alist = [chr(i) for i in range(65, 65+26)]
    a3_dic = {}
    a3_dic["<pad>"] = 0
    a3_dic["<cls>"] = 1
    a3_dic["<eos>"] = 2
    a3_dic["<unk>"] = 3

    if not cls_eos:
        a3_dic.pop("<cls>")
        a3_dic.pop("<eos>")
        a3_dic["<unk>"] = 1

    num = len(a3_dic)
    if not ex_aa:
        ex_AAs = set(["B", "J", "O", "U", "X", "Z"])
        alist = [i for i in alist if i not in ex_AAs]
    
    for aa in alist:
        a3_dic[aa] = num
        num += 1
    
    return a3_dic