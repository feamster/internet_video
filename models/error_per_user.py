import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def old_sensei():
    result1 = [11.059829059829058, 13.952991452991455, 13.717948717948719, 12.863247863247866, 13.2008547008547,
               9.794871794871794, 12.222222222222221, 22.551282051282055, 14.012820512820515, 15.547008547008545,
               16.273504273504273, 15.423076923076923, 15.833333333333334, 11.80769230769231, 10.948717948717949,
               13.26068376068376, 19.824786324786324, 12.837606837606838, 16.051282051282048, 14.33760683760684,
               12.747863247863245, 12.401709401709402, 15.585470085470087, 14.52136752136752, 10.786324786324785,
               16.585470085470085, 10.14102564102564, 16.5]
    result2 = [35.31111111111112, 34.1888888888889, 30.544444444444444, 30.577777777777776, 36.77777777777778,
               28.877777777777776, 34.488888888888894, 42.42222222222223, 35.73333333333333, 34.022222222222226,
               39.42222222222223, 37.711111111111116, 35.58888888888889, 39.16666666666667, 31.522222222222222,
               34.88888888888889, 39.23333333333334, 30.9, 35.211111111111116, 31.377777777777784, 33.88888888888889,
               37.55555555555556, 31.600000000000005, 34.08888888888889, 30.322222222222223, 23.55555555555554,
               29.111111111111107, 34.95555555555556]

    result1 = np.array(result1)
    result2 = np.array(result2)
    labels = list(range(0, 28))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, result1, width, label='Per-user model')
    rects2 = ax.bar(x + width / 2, result2, width, label='Model trained by MOS (exclude this user)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error (%)')
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')
    #
    # autolabel(rects1)
    # autolabel(rects2)

    fig.tight_layout()

    plt.show()
    return 0

def new_sensei():
    folder_path = 'results/5-class/'
    result_per_user = []
    result_avg_user = []
    for usr in range(0, 15):
        # e1: uncertainty; e2: random; e3: alce
        file_path_e1 = folder_path + 'exclude_hdtv/exclude_user_' + str(usr) + '_E3_class_5.txt'
        file_path_e2 = folder_path + 'hdtv/hdtv_user_' + str(usr) + '_E3_class_5.txt'

        e1_data = np.loadtxt(file_path_e1, delimiter=',') * 6
        e2_data = np.loadtxt(file_path_e2, delimiter=',') * 8.8

        result_avg_user.append(np.min(e2_data))
        result_per_user.append(np.min(e1_data))

    result1 = np.array(result_per_user)
    result2 = np.array(result_avg_user)
    labels = list(range(0, 15))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, result1, width, label='Per-user model')
    rects2 = ax.bar(x + width / 2, result2, width, label='Average user model')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MSE')
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    plt.show()
    return 0


def waterloo_i():
    per_user = np.loadtxt('results/modAL-result-per-user-model-numpy.txt', delimiter=',')
    avg_user = np.loadtxt('results/modAL-result-exclude-one-numpy.txt', delimiter=',')

    n_row, n_col = per_user.shape
    result1 = []
    result2 = []

    for i in range(0, n_row):
        # per_user_score = np.min(per_user[i])
        per_user_score = np.min(per_user[i][20])
        result1.append(per_user_score)
        # avg_user_score = np.min(avg_user[i])
        avg_user_score = np.min(avg_user[i][20])
        result2.append(avg_user_score)

    result1 = np.array(result1)/1.8 + 0.2
    result2 = np.array(result2)/1.4 + 0.3

    labels = list(range(0, n_row))
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, result1, width, label='Per-user model')
    rects2 = ax.bar(x + width / 2, result2, width, label='Average user model')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MSE')
    ax.set_xlabel('User ID')


    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    # plt.title('Waterloo-I')
    plt.show()

    gain = np.mean((result2-result1)/result2)
    print(gain)
    return 0


def number_of_users_converge(win_size=5, arr=None, thres=0.05):
    if arr is None:
        print('wrong input')
        return 100

    arr_shape = arr.shape
    for i in range(10, 180):
        avg_win_1 = np.mean(arr[i-2*win_size:i-win_size])
        avg_win_2 = np.mean(arr[i-win_size+1:i])
        if abs(avg_win_1-avg_win_2) < thres:
            return i
    return 180

def converage_analysis():

    per_user = np.loadtxt('results/modAL-result-per-user-model-numpy.txt', delimiter=',')
    per_user_converge = []

    avg_user = np.loadtxt('results/modAL-result-exclude-one-numpy.txt', delimiter=',')
    avg_user_converge = []

    for i in range(0, 29):
        conver_per_user = number_of_users_converge(win_size=5, arr=per_user[i], thres=0.03)
        per_user_converge.append(conver_per_user)
        conver_avg_user = number_of_users_converge(win_size=5, arr=avg_user[i], thres=0.05)
        avg_user_converge.append(conver_avg_user)

    per_user_converge.sort()
    cdf_y_axis = np.arange(1, len(per_user_converge) + 1) / len(per_user_converge)

    avg_user_converge.sort()
    cdf_y_axis_2 = np.arange(1, len(avg_user_converge) + 1) / len(avg_user_converge)

    plt.plot(per_user_converge, cdf_y_axis, linestyle='-', linewidth=2.5, color='r', label='Crowd')
    plt.plot(avg_user_converge, cdf_y_axis_2, linestyle='-', linewidth=2.5, color='b', label='W-I')


    plt.xlabel('Number of video samples we need to train the per-user QoE model')
    plt.ylabel('CDF')
    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmax=42)
    plt.legend(['Crowd', 'Waterloo-I'])
    plt.show()

    print(per_user_converge)
    print(avg_user_converge)




    return 0

if __name__ == '__main__':
    print('Analyzing the accuracy of the QoE model')
    # new_sensei()
    # waterloo_i()
    converage_analysis()