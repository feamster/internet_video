import os
import sys
import numpy as np
import pandas as pd
import osqp as osp
from scipy.sparse import vstack
from scipy.sparse import csc_matrix

# private class
import waterloo_iv_processing

eps = 1e-3
n_chunk = 7
n_class = 5


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


# obtain the position in the stall matrix
def get_s_idx(p, t):
    '''
    :param p: the stall time for the previous chunk
    :param t: the stall time for the current chunk
    :return: the matrix idx we should retrieve
    '''
    # -eps to quantize 100 into the last bin
    # Here, 10 is the size of the matrix. Current matrix size: 10 x 10
    si = int(max((p - eps), 0) / 10)
    sj = int(min(max(round(t - eps), 0), 10)) - 1
    return si, sj


# obtain the position in the bitrate switch matrix
def get_a_idx(p, delta_p):
    '''
    :param p: the bitrate (vmaf) score of the previous chunk
    :param delta_p: the delta bitrate (vmaf) score
    :return: the matrix idx we should retrieve
    '''
    # -eps to quantize 100 into the last bin
    ai = int(max((p - eps), 0) / 10)
    # handle \Delta p = 95~100 and -100~-95
    aj = int(min(max(round(delta_p / 10), -10 + 1), 10 - 1))
    aj += ai
    aj = min(max(aj, 0), 10 - 1)
    return ai, aj


def test(X, y):
    # stall penalty matrix
    p_s = np.loadtxt('utils/ksqi/s_model.txt')
    # bitrate switch penalty matrix
    p_a = np.loadtxt('utils/ksqi/a_model.txt')

    predict_score = []
    for i in range(len(X)):
        cum_qoe = []
        is_first = True
        sample_vmaf = X[i, 0:n_chunk]
        sample_rebuf = X[i, n_chunk:14]
        prev_stall = 0
        cur_stall = 0
        prev_vmaf = 0
        cur_vmaf = 0
        for c_id in range(0, n_chunk):
            penalty_bit_switch = 0
            penalty_stall = 0
            cur_stall = sample_rebuf[c_id]
            cur_vmaf = sample_vmaf[c_id]
            s_i = 0
            s_j = 0
            a_i = 0
            a_j = 0
            if is_first:
                is_first = False
                si, sj = get_s_idx(80, cur_stall / 9)
            else:
                si, sj = get_s_idx(prev_vmaf, cur_stall)
                ai, aj = get_a_idx(prev_vmaf, cur_vmaf - prev_vmaf)
                penalty_bit_switch = p_a[ai, aj]
            # handle stalling duration = 0
            if sj == -1:
                penalty_stall = 0
            else:
                penalty_stall = p_s[si, sj]

            cum_qoe.append(cur_vmaf + penalty_stall + penalty_bit_switch)
            prev_stall = cur_stall
            prev_vmaf = cur_vmaf
        cum_qoe = np.array(cum_qoe)
        predict_score.append(np.mean(cum_qoe))

    y = np.array(y)
    predict_score = np.array(predict_score)
    y = map_to_class(y=y, n_class=n_class)
    predict_score = map_to_class(y=predict_score, n_class=n_class)
    y = np.array(y)
    predict_score = np.array(predict_score)

    average_error = np.mean(np.abs(y - predict_score))/n_class

    return average_error


def sys_main():
    print('Processing the input data')
    for user_id in range(0, 28):
        video_score, video_quality = waterloo_iv_processing.get_non_normalized_score_data(device='hdtv',video_name=['sports', 'document', 'nature', 'game', 'movie'])
        X = video_quality
        y = video_score[:, user_id]
        avg_error = test(X=X, y=y)
        print(avg_error)

    return 0


if __name__ == '__main__':
    print('KSQI model test')
    sys_main()
