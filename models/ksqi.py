import os
import sys
import numpy as np
import pandas as pd
import osqp as osp
from scipy.sparse import vstack
from scipy.sparse import csc_matrix


def normalized_scores(score_arr):
    score_arr = np.array(score_arr)
    t_scores = score_arr - np.min(score_arr)
    t_scores = t_scores / np.max(t_scores)
    score_arr = t_scores
    return score_arr


def map_to_class(y=None, n_class=10):
    if y is None:
        print('No available data points')
        return 0
    y = normalized_scores(score_arr=y)
    unit_class = 1.0 / n_class
    Y = []
    for i in range(0, len(y)):
        this_class = int(y[i] / unit_class)
        if this_class >= n_class:
            this_class = n_class - 1
        Y.append(this_class)
    return Y


def test(X, y):
    # stall matrix
    p_s = np.loadtxt('models/utils/ksqi/s_model.txt')
    # bitrate switch matrix
    p_a = np.loadtxt('models/utils/ksqi/a_model.txt')

    n_chunk = 7
    n_class = 10
    predict_score = []
    for i in range(len(X)):
        cum_qoe = 0
        is_first = True
        sample_vmaf = X[i, 0:n_chunk]
        sample_rebuf = X[i, n_chunk:14]
        prev_stall = 0
        cur_stall = 0
        prev_vmaf = 0
        cur_vmaf = 0
        for c_id in range(0, n_chunk):
            if is_first:
                cum_qoe =




    return 0


def sys_main():
    print()
    return 0


if __name__ == '__main__':
    sys_main()
