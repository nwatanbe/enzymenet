#!/bin/bash
python -m main_vv select_samples_for_vec_ref.fasta select_samples_for_vec_search.fasta 1 --select_num=10 --n_neighbors=15 --metric="euclidean"