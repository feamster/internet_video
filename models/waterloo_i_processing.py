import pandas as pd
import numpy as np
import os

name = {}
name_cnt = 1

def normalized_scores(scores):
    scores = np.array(scores)
    row, col = scores.shape
    for i in range(0, col):
        t_scores = scores[:, i]
        t_scores = t_scores - np.min(t_scores)
        t_scores = t_scores / np.max(t_scores)
        scores[:, i] = t_scores
    return scores


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

def get_per_user_data(user_id=None):

    user_id += 1

    
    if (user_id is None):
        print('Input should not have null parameters.')
        return None

    all_data = pd.read_csv('open_dataset/waterloo_dataset/WaterlooSQoE-I/data.csv', header = None)
    mos_data = pd.read_csv('open_dataset/waterloo_dataset/WaterlooSQoE-I/mos.csv', header=None)

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
        if abs(float(mos_data[2][i+1])) < 1e-6: continue
        video_name.append(process_video_name(all_data[i][0]))
        user_scores.append(int(all_data[i][user_id]))
        mos_scores.append(float(mos_data[1][i+1]))
        mssim_scores.append(float(mos_data[2][i+1]))
        psnr_scores.append(float(mos_data[3][i+1]))
        ssim_scores.append(float(mos_data[4][i+1]))
        ssimplus_scores.append(float(mos_data[5][i+1]))

        mssim_smooth.append(float(mos_data[6][i+1]))
        psnr_smooth.append(float(mos_data[7][i+1]))
        ssim_smooth.append(float(mos_data[8][i+1]))
        ssimplus_smooth.append(float(mos_data[9][i+1]))


    
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

        ret_data_row = [usr_score, psnr, psnr_s] 
        # ret_data_row.extend(video_name[i])
        ret_data.append(ret_data_row)

    return ret_data

def get_non_normalized_score_data(device=None, video_name=None):
    if (device is None) or (video_name is None):
        print('Input should not have null parameters.')
        return None

    df_data = pd.read_csv('open_dataset/waterloo_dataset/WaterlooSQoE-IV/data.csv', usecols=['streaming_log', 'device', 'content', 'individual_scores', 'mos'])
    grouped = df_data.groupby(['device', 'content'])
    video_scores = []
    streaming_log = []

    for name, group in grouped:
        if (name[0] == device) and (name[1] in video_name):
            user_score_str_list = list(group['individual_scores'])
            streaming_log.extend(list(group['streaming_log']))
            mos_list = list(group['mos'])
            for idx in range(0, len(user_score_str_list)):
                x = user_score_str_list[idx]
                user_score_int_list = x.strip('][').split(' ')
                user_score_arr = list(map(float, user_score_int_list))
                user_score_arr.append(mos_list[idx])
                video_scores.append(user_score_arr)

    video_scores = np.array(video_scores)
    print('Size of all user data:', video_scores.shape)

    # ret_data format: vmaf ('vmaf'), rebuffering time ('rebuffering_duration')
    ret_data = []
    for i in range(0, len(streaming_log)):
        log_name = streaming_log[i]
        f_name = 'open_dataset/waterloo_dataset/WaterlooSQoE-IV/streaming_logs/' + log_name
        df_log = pd.read_csv(f_name, usecols=['vmaf', 'rebuffering_duration'])
        rebuf_time = df_log['rebuffering_duration'].tolist()
        vmaf = df_log['vmaf'].tolist()
        # col1 = 
        ret_data_row = []
        ret_data_row.extend(vmaf)
        ret_data_row.extend(rebuf_time)
        ret_data.append(ret_data_row)

    ret_data = np.array(ret_data)

    return video_scores, ret_data


def ex_score(user_id, score_arr, video_id):
    new_socre = np.delete(score_arr, user_id, 1)
    ret_data = np.mean(new_socre[video_id, :])
    return ret_data


def get_all_but_one_user(ex_user_id=None, device=None, video_name=None):

    user_id = ex_user_id + 1
    
    if (user_id is None):
        print('Input should not have null parameters.')
        return None

    all_data = pd.read_csv('open_dataset/waterloo_dataset/WaterlooSQoE-I/data.csv', header = None)
    mos_data = pd.read_csv('open_dataset/waterloo_dataset/WaterlooSQoE-I/mos.csv', header=None)

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

    temp_scores = {}
    for i in range(30):
        temp_scores[i] = []

    for i in range(all_data.shape[1]):
        if abs(float(mos_data[2][i+1])) < 1e-6: continue
        video_name.append(process_video_name(all_data[i][0]))

        for j in range(30):
            temp_scores[j].append(int(all_data[i][j+1]))

        user_scores.append(int(all_data[i][user_id]))
        mos_scores.append(float(mos_data[1][i+1]))
        mssim_scores.append(float(mos_data[2][i+1]))
        psnr_scores.append(float(mos_data[3][i+1]))
        ssim_scores.append(float(mos_data[4][i+1]))
        ssimplus_scores.append(float(mos_data[5][i+1]))

        mssim_smooth.append(float(mos_data[6][i+1]))
        psnr_smooth.append(float(mos_data[7][i+1]))
        ssim_smooth.append(float(mos_data[8][i+1]))
        ssimplus_smooth.append(float(mos_data[9][i+1]))
    
    for i in range(30):
        temp_scores[i] = normalize(temp_scores[i])
    user_scores = normalize(user_scores)

    ex_scores = []

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

    for i in range(valid_data_cnt):
        temp = 0.0
        for j in range(30): temp+=temp_scores[j][i]
        temp -= user_scores[i]
        temp /= 29
        ex_scores.append(temp)

    ex_scores = normalize(ex_scores)

    ret_data = []
    for i in range(valid_data_cnt):
        usr_score = user_scores[i]
        ex_usr_score = ex_scores[i]

        mos = mos_scores[i]
        mssim = mssim_scores[i]
        psnr = psnr_scores[i]
        ssim = ssim_scores[i]
        ssimplus = ssimplus_scores[i]

        mssim_s = mssim_smooth[i]
        psnr_s = psnr_smooth[i]
        ssim_s = ssim_smooth[i]
        ssimplus_s = ssimplus_smooth[i]

        ret_data_row = [usr_score, psnr, psnr_s] 

        # ret_data_row = [usr_score, psnr, psnr_s] !! 
        # ret_data_row.extend(video_name[i])
        ret_data.append(ret_data_row)

    return ret_data

def sys_main():
    # video_content_arr = ['sports', 'document', 'nature', 'game', 'movie']
    # device_type_arr = ['hdtv', 'uhdtv', 'phone']
    test_data = get_per_user_data(user_id=1)#, device='hdtv', video_name=['sports'])

    # print(test_data)
    return 0


if __name__ == '__main__':
    print('Processing waterloo-I')
    sys_main()