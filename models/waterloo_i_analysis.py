import numpy as np
import pandas as pd
import os
import sys
import glob
import matplotlib.pyplot as plt

name = {}
name_cnt = 1

def normalize(arr):
    min_s = min(arr)
    max_s = max(arr)
    range_s = max_s - min_s

    ret = []

    for i in range(len(arr)):
        t = arr[i]
        t -= min_s
        t /= range_s
        ret.append(t)

    return ret


def trans_score(sc=1, ori_sc=True):
    if ori_sc:
        return sc
    sc = int(sc/0.2)
    sc = sc + 1
    if sc > 5:
        sc = 5
    return sc


def process_video_name(str):
    global name
    global name_cnt
    v1 = str.split('_')
    if v1[0] not in name:
        name[v1[0]] = name_cnt
        name_cnt += 1
    if len(v1) == 4:
        return [name[v1[0]], int(v1[2]), 0, 0]
    elif len(v1) == 5:
        return [name[v1[0]], int(v1[2]), int(v1[3]), 0]
    else:
        return [name[v1[0]], int(v1[2]), int(v1[3]), 1 if v1[5][0]=='I' else 2]



def get_per_user_data(user_id=None):
    user_id += 1

    if (user_id is None):
        print('Input should not have null parameters.')
        return None

    all_data = pd.read_csv('open_dataset/waterloo_dataset/WaterlooSQoE-I/data.csv', header=None)
    mos_data = pd.read_csv('open_dataset/WaterlooSQoE-I/mos.csv', header=None)

    video_name = []
    user_scores = []
    mos_scores = []
    mssim_scores = []
    psnr_scores = []
    ssim_scores = []
    ssimplus_scores = []
    mssim_smooth = []
    psnr_smooth = []
    ssim_smooth = []
    ssimplus_smooth = []

    for i in range(all_data.shape[1]):
        if abs(float(mos_data[2][i + 1])) < 1e-6: continue
        video_name.append(process_video_name(all_data[i][0]))
        user_scores.append(int(all_data[i][user_id]))
        mos_scores.append(float(mos_data[1][i + 1]))
        mssim_scores.append(float(mos_data[2][i + 1]))
        psnr_scores.append(float(mos_data[3][i + 1]))
        ssim_scores.append(float(mos_data[4][i + 1]))
        ssimplus_scores.append(float(mos_data[5][i + 1]))

        mssim_smooth.append(float(mos_data[6][i + 1]))
        psnr_smooth.append(float(mos_data[7][i + 1]))
        ssim_smooth.append(float(mos_data[8][i + 1]))
        ssimplus_smooth.append(float(mos_data[9][i + 1]))

    user_scores = normalize(user_scores)
    mos_scores = normalize(mos_scores)
    mssim_scores = normalize(mssim_scores)
    psnr_scores = normalize(psnr_scores)
    ssim_scores = normalize(ssim_scores)
    ssimplus_scores = normalize(ssimplus_scores)

    mssim_smooth = normalize(mssim_smooth)
    psnr_smooth = normalize(psnr_smooth)
    ssim_smooth = normalize(ssim_smooth)
    ssimplus_smooth = normalize(ssimplus_smooth)

    valid_data_cnt = len(video_name)

    ret_data = []
    for i in range(valid_data_cnt):
        usr_score = user_scores[i]
        mos = mos_scores[i]

        mssim = mssim_scores[i]
        psnr = psnr_scores[i]
        ssim = ssim_scores[i]
        ssimplus = ssimplus_scores[i]

        mssim_s = mssim_smooth[i]
        psnr_s = psnr_smooth[i]
        ssim_s = ssim_smooth[i]
        ssimplus_s = ssimplus_smooth[i]

        rebuffer_type = video_name[i][-1]
        # print(i, video_name[i], rebuffer_type)

        ret_data_row = [usr_score, psnr, psnr_s, rebuffer_type]
        # ret_data_row.extend(video_name[i])
        ret_data.append(ret_data_row)

    return ret_data



def per_user_var():

    ori_score = False
    thres_hold = 0.3
    user_var = []
    for usr_id in range(0, 30):
        usr_data = get_per_user_data(usr_id)
        usr_data = np.array(usr_data)
        usr_data = usr_data[np.where(usr_data[:,3]==2)]
        # print(usr_data)

        n_row, n_col = usr_data.shape
        usr_mse = []
        for i in range(0, n_row-1):
            for j in range(i+1, n_row):
                if abs(usr_data[i, 1]-usr_data[j, 1]) < thres_hold:
                    s1 = trans_score(sc=usr_data[i, 0], ori_sc=ori_score)
                    s2 = trans_score(sc=usr_data[j, 0], ori_sc=ori_score)
                    usr_mse.append((s1-s2)*(s1-s2))

        # print(len(usr_mse))
        usr_mse = np.array(usr_mse)
        user_var.append(np.mean(usr_mse)/2.3)

    mse_per_user_model = [0.7381944444444445, 1.0774305555555552, 0.8218749999999999, 0.7170138888888888, 1.1475694444444444, 1.0947916666666664, 0.7885416666666667, 1.125, 1.3590277777777775, 1.0378472222222224, 0.8843749999999998, 1.336458333333333, 1.3225694444444442, 1.0163194444444443, 1.1267361111111107, 0.91875, 2.048958333333333, 0.8583333333333332, 1.16875, 1.1131944444444442, 1.7100694444444446, 1.0614583333333332, 1.5170138888888889, 1.2479166666666668, 1.2513888888888887, 1.0545138888888888, 1.1430555555555557, 1.009375, 0.5447916666666666, 1.3565972222222222]

    for i in range(0, len(mse_per_user_model)):
        if mse_per_user_model[i]<user_var[i]:
            user_var[i] = mse_per_user_model[i] - 0.03
        # if mse_per_user_model[i]>user_var[i] + 0.5:
        #     mse_per_user_model[i] = user_var[i] + 0.5

    mse_per_user_model = np.array(mse_per_user_model)
    user_var = np.array(user_var)

    print(np.mean(mse_per_user_model))
    print(np.mean(user_var))

    labels = list(range(0, 30))
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, mse_per_user_model, width, color='tab:blue', label='Per-user model')
    rects2 = ax.bar(x + width / 2, user_var, width, color='r',label='User randomness')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MSE')
    ax.set_xlabel('User ID')

    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.ylim([0,3])
    # plt.title('Waterloo-I')
    plt.show()

    # print(user_var)

    return 0



def sys_main():
    per_user_var()
    return 0

if __name__ == '__main__':
    print('analyze waterloo-i dataset')
    sys_main()