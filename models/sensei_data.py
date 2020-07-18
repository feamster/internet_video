import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def intersect_video(video1, video2):
    return_set = list(set(video1).intersection(set(video2)))
    return return_set

def meta_read():
    df_data = pd.read_csv('sensei/user_scores.csv', usecols=['campaign_name', 'num_chunk', 'mturk_id', 'accept'])
    df_data = df_data[df_data['accept'] > 0]

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
    count_stats = meta_read()
    counts = list(count_stats.values())
    counts.sort(reverse=True)
    x_label = range(0, len(counts))
    counts = np.array(counts) * 5
    plt.bar(x_label, counts)
    plt.xlabel('Turker ID')
    plt.ylabel('# of rated videos')
    plt.show()
    return 0


if __name__ == '__main__':
    print('Use sensei data')
    sys_main()
