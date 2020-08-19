import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def intersect_video(video1, video2):
    return_set = list(set(video1).intersection(set(video2)))
    return return_set


def extract_data(usr_id=None):
    df_data = pd.read_csv('sensei/user_scores.csv', usecols=['campaign_name', 'num_chunk', 'mturk_id', 'accept', 'chunk1', 'chunk2', 'chunk3', 'chunk4', 'chunk5'])
    df_data = df_data[df_data['accept'] >= 0]
    df_data = df_data[df_data['num_chunk'] ==5]
    df_data = df_data[df_data['mturk_id'] ==usr_id]
    X = []
    y = []
    ret_data = []
    y = []

    for index, row in df_data.iterrows():
        v_name = row['campaign_name']
        for i in range(0, 5):
            filename = 'sensei/log/' + v_name + '_' + str(i) + '.csv'
            df_log = pd.read_csv(filename, usecols=['vmaf', 'rebuf'])
            rebuf_time = df_log['rebuf'].tolist()
            vmaf = df_log['vmaf'].tolist()
            ret_data_row = []
            ret_data_row.extend(vmaf)
            ret_data_row.extend(rebuf_time)
            ret_data.append(ret_data_row)
            row_name = 'chunk'+str(i+1)
            y.append(row[row_name])

    ret_data = np.array(ret_data)
    y = np.array(y)
    return ret_data, y


def reprocess_data():
    df_data = pd.read_csv('sensei/user_scores.csv', usecols=['campaign_name', 'num_chunk', 'mturk_id', 'accept', 'chunk1', 'chunk2', 'chunk3', 'chunk4', 'chunk5'])
    df_data = df_data[df_data['accept'] >= 0]
    df_data = df_data[df_data['num_chunk'] ==5]

    video_quality = []
    video_rebuf = []
    video_content = []
    for index, row in df_data.iterrows():
        video_str_arr = row['campaign_name'].split('_')
        video_content.append(video_str_arr[0])
        video_rebuf.append(int(video_str_arr[1]))
        v_q = int(video_str_arr[2].strip('k'))
        if (v_q == 3000) or (v_q==1000)  or (v_q == 2000):
            v_q = 3
        elif (v_q == 300) or (v_q==500):
            v_q = 2
        else:
            v_q = 1
        video_quality.append(v_q)

    df_data['video_name'] = video_content
    df_data['video_bitrate'] = video_quality
    df_data['video_rebuf'] = video_rebuf

    df_data.to_csv('sensei/user_scores_1.csv', index=False)

    grouped = df_data.groupby(['campaign_name'])

    for name, group in grouped:
        v_q = list(group['video_bitrate'])[0]
        v_b = list(group['video_rebuf'])[0]
        for i in range(0, 5):
            filename = name + '_' + str(i)
            save_df = {'vmaf':[], 'rebuf':[]}
            bit_list = [3]*5
            buf_list = [0]*5
            bit_list[i] = v_q
            buf_list[i] = v_b
            save_df['vmaf'] = bit_list
            save_df['rebuf'] = buf_list
            save_df = pd.DataFrame(data=save_df)
            save_df.to_csv('sensei/log/'+filename+'.csv', index=False)
    return 0


def reprocess_data_2():
    df_data = pd.read_csv('sensei/user_scores.csv', usecols=['campaign_name', 'num_chunk', 'mturk_id', 'accept', 'chunk1', 'chunk2', 'chunk3', 'chunk4', 'chunk5', 'video_name', 'video_bitrate', 'video_rebuf'])
    df_data = df_data[df_data['accept'] >= 0]
    df_data = df_data[df_data['num_chunk'] ==5]

    video_quality = []
    video_rebuf = []
    video_content = []
    for index, row in df_data.iterrows():
        video_str_arr = row['campaign_name'].split('_')
        video_content.append(video_str_arr[0])
        video_rebuf.append(int(video_str_arr[1]))
        video_quality.append(int(video_str_arr[2].strip('k')))

    df_data['video_name'] = video_content
    df_data['video_bitrate'] = video_quality
    df_data['video_rebuf'] = video_rebuf

    df_data.to_csv('sensei/user_scores_1.csv', index=False)

    return 0


def meta_read():
    df_data = pd.read_csv('sensei/user_scores.csv', usecols=['campaign_name', 'num_chunk', 'mturk_id', 'accept'])
    df_data = df_data[df_data['accept'] >= 0]
    df_data = df_data[df_data['num_chunk'] ==5]

    grouped = df_data.groupby(['mturk_id'])
    count_stats = {}

    turker_id = []
    turker_count = []
    turker_videos = []

    for name, group in grouped:
        video_name = list(group['campaign_name'])
        count_stats[name] = len(video_name)
        turker_id.append(name)
        turker_count.append(len(video_name))
        turker_videos.append(video_name)

    zipped = zip(turker_id, turker_count, turker_videos)
    sort_zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    result = zip(*sort_zipped)
    turker_id, turker_count, turker_videos = [list(x) for x in result]

    video_set = turker_videos[0]
    num_of_intersect = [len(video_set)]

    for i in range(1, 30):
        temp_video = turker_videos[i]
        inter_sec = intersect_video(video_set, temp_video)
        num_of_intersect.append(len(inter_sec))
        video_set = inter_sec

    x_label = range(0, len(num_of_intersect))
    num_of_intersect = np.array(num_of_intersect) * 5
    plt.bar(x_label, num_of_intersect)
    plt.xlabel('Top k turkers')
    plt.ylabel('# of common videos for top k turkers')
    plt.show()

    return count_stats



def sys_main():
    # count_stats = meta_read()
    # counts = list(count_stats.values())
    # counts.sort(reverse=True)
    # x_label = range(0, len(counts))
    # counts = np.array(counts) * 5
    # plt.bar(x_label, counts)
    # plt.xlabel('Turker ID')
    # plt.ylabel('# of rated videos')
    # plt.show()
    # reprocess_data()
    a, b = extract_data('A3P2WQO8VESWF2')
    print(a)
    print(b)
    return 0


if __name__ == '__main__':
    print('Use sensei data')
    sys_main()
