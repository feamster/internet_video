import pandas as pd
import numpy as np
import os


def normalized_scores(scores):
    scores = np.array(scores)
    row, col = scores.shape
    for i in range(0, col):
        t_scores = scores[:, i]
        t_scores = t_scores - np.min(t_scores)
        t_scores = t_scores / np.max(t_scores)
        scores[:, i] = t_scores
    return scores


def get_per_user_data(user_id=None, device=None, video_name=None):
    if (user_id is None) or (device is None) or (video_name is None):
        print('Input should not have null parameters.')
        return None

    df_data = pd.read_csv('open_dataset/waterloo_dataset/WaterlooSQoE-IV/data.csv', usecols=['streaming_log', 'device', 'content', 'individual_scores'])
    grouped = df_data.groupby(['device', 'content'])
    video_scores = []
    streaming_log = []

    for name, group in grouped:
        if (name[0] == device) and (name[1] in video_name):
            user_score_str_list = list(group['individual_scores'])
            streaming_log.extend(list(group['streaming_log']))
            for x in user_score_str_list:
                user_score_int_list = x.strip('][').split(' ')
                video_scores.append(list(map(float, user_score_int_list)))

    video_scores = normalized_scores(video_scores)
    print('Size of all user data:', video_scores.shape)

    # ret_data format: usr score, vmaf ('vmaf'), rebuffering time ('rebuffering_duration')
    ret_data = []
    for i in range(0, len(streaming_log)):
        log_name = streaming_log[i]
        f_name = 'open_dataset/waterloo_dataset/WaterlooSQoE-IV/streaming_logs/' + log_name
        df_log = pd.read_csv(f_name, usecols=['vmaf', 'rebuffering_duration'])
        usr_score = video_scores[i, user_id]
        rebuf_time = df_log['rebuffering_duration'].tolist()
        vmaf = df_log['vmaf'].tolist()
        ret_data_row = [usr_score]
        ret_data_row.extend(vmaf)
        ret_data_row.extend(rebuf_time)
        ret_data.append(ret_data_row)

    return ret_data


def ex_score(user_id, score_arr, video_id):

    new_socre = np.delete(score_arr, user_id, 1)
    ret_data = np.mean(new_socre[video_id, :])
    return ret_data


def get_all_but_one_user(ex_user_id=None, device=None, video_name=None):

    if (ex_user_id is None) or (device is None) or (video_name is None):
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

                n_user = len(user_score_arr)
                exclude_mos = (mos_list[idx]*n_user - user_score_arr[ex_user_id])/(n_user-1)
                user_score_arr.append(exclude_mos)
                video_scores.append(user_score_arr)

    video_scores = normalized_scores(video_scores)

    print('Size of all user data:', video_scores.shape)

    # ret_data format: usr score, vmaf ('vmaf'), rebuffering time ('rebuffering_duration')
    ret_data = []
    for i in range(0, len(streaming_log)):
        log_name = streaming_log[i]
        f_name = 'open_dataset/waterloo_dataset/WaterlooSQoE-IV/streaming_logs/' + log_name
        df_log = pd.read_csv(f_name, usecols=['vmaf', 'rebuffering_duration'])

        usr_score = video_scores[i, ex_user_id]
        # mos_score = ex_score(user_id=ex_user_id, score_arr=video_scores, video_id=i)
        mos_score = video_scores[i, -1]
        rebuf_time = df_log['rebuffering_duration'].tolist()
        vmaf = df_log['vmaf'].tolist()
        ret_data_row = [mos_score]
        ret_data_row.append(usr_score)
        ret_data_row.extend(vmaf)
        ret_data_row.extend(rebuf_time)
        ret_data.append(ret_data_row)

    return ret_data


def sys_main():
    video_content_arr = ['sports', 'document', 'nature', 'game', 'movie']
    device_type_arr = ['hdtv', 'uhdtv', 'phone']
    test_data = get_per_user_data(user_id=1, device='hdtv', video_name=['sports'])

    print(test_data)
    return 0


if __name__ == '__main__':
    print('Processing waterloo-IV')
    sys_main()