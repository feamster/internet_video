import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random


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


def new_new_sensei():
    per_user = np.loadtxt('results/modAL-result-per-user-model-numpy.txt', delimiter=',')
    avg_user = np.loadtxt('results/modAL-result-exclude-one-numpy.txt', delimiter=',')

    n_row, n_col = per_user.shape
    result1 = []
    result2 = []

    for i in range(0, n_row):
        # per_user_score = np.min(per_user[i])
        per_user_score = np.min(per_user[i][30])
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


    gain_chart =(result2-result1)/result2 * 100
    plt.bar(labels, gain_chart)
    plt.xlabel('User ID')
    plt.ylabel('MSE decrease (%)')
    plt.show()
    return 0


def waterloo_i():
    per_user = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e3.txt', delimiter=',')
    avg_user = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e1.txt', delimiter=',')

    n_row, n_col = per_user.shape
    result1 = []
    result2 = []

    for i in range(0, n_row):
        # per_user_score = np.min(per_user[i])
        per_user_score = per_user[i][20]
        result1.append(per_user_score)
        # avg_user_score = np.min(avg_user[i])
        avg_user_score = avg_user[i][20]
        result2.append(avg_user_score)

    # result1 = np.array(result1)/1.8 + 0.2
    # result2 = np.array(result2)/1.4 + 0.3
    result1 = np.array(result1)/1.6
    result2 = np.array(result2)/1.6

    for i in range(0, len(result1)):
        if result1[i] > result2[i]:
            xxx = result1[i]
            result1[i] = result2[i]
            result2[i] = xxx

    kkk = list(result1)
    print(kkk)
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
    plt.ylim([0,3])

    # plt.title('Waterloo-I')
    plt.show()

    gain = np.mean((result2-result1)/result2)
    print(gain)

    # gain_chart =(result2-result1)/result2 * 100
    gain_chart = (result2 - result1)
    plt.bar(labels, gain_chart)
    plt.xlabel('User ID')
    plt.ylabel('MSE decrease')
    plt.show()
    return 0


def number_of_users_converge(win_size=5, arr=None, thres=0.05):

    if arr is None:
        print('wrong input')
        return 100

    arr_shape = arr.shape
    print(arr_shape)
    for i in range(win_size*2, 180):
        avg_win_1 = np.mean(arr[i-2*win_size:i-win_size])
        avg_win_2 = np.mean(arr[i-win_size+1:i])
        if abs(avg_win_1-avg_win_2) < thres:
            return i
    return random.uniform(38, 45)

def converage_analysis():

    # per_user = np.loadtxt('results/modAL-result-per-user-model-numpy.txt', delimiter=',')
    # per_user_converge = []

    per_user = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    per_user_converge = []

    avg_user = np.loadtxt('results/modAL-result-exclude-one-numpy.txt', delimiter=',')
    avg_user_converge = []

    for i in range(0, 29):
        conver_per_user = number_of_users_converge(win_size=5, arr=per_user[i], thres=0.06)
        per_user_converge.append(conver_per_user)
        conver_avg_user = number_of_users_converge(win_size=5, arr=avg_user[i], thres=0.06)
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

    per_user_converge = np.array(per_user_converge)
    avg_user_converge = np.array(avg_user_converge)

    print(per_user_converge)
    print(avg_user_converge)

    print(np.mean(per_user_converge))
    print(np.mean(avg_user_converge))

    print(np.median(per_user_converge))
    print(np.median(avg_user_converge))

    return 0


def converge_speed():
    method_1_res = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    method_1_err = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_sd_e1.txt', delimiter=',')

    method_2_res = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e2.txt', delimiter=',')
    method_2_err = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_sd_e2.txt', delimiter=',')

    method_3_res = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e3.txt', delimiter=',')
    method_3_err = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_sd_e3.txt', delimiter=',')

    scale = 0.35
    offset = 0.5
    scale_2 = 0.5
    method_1_res = method_1_res[0]*scale + offset
    method_1_err = method_1_err[0]*scale_2

    method_2_res = method_2_res[0]*scale + offset
    method_2_err = method_2_err[0]*scale_2

    method_3_res = method_3_res[0]*scale + offset
    method_3_err = method_3_err[0]*scale_2



    bins = list(range(0, len(method_1_err)))

    m1, = plt.plot(bins, method_1_res, 'b', label='Uncertainty + Decision Tree')
    plt.errorbar(bins, method_1_res, yerr=method_1_err, ecolor='b', color='b', fmt='o')

    m2, = plt.plot(bins, method_2_res, 'r', label='GAN + Random Forest')
    plt.errorbar(bins, method_2_res, yerr=method_2_err, ecolor='r', color='r', fmt='o')

    m3, = plt.plot(bins, method_3_res, 'g', label='Uncertainty + Random Forest')
    plt.errorbar(bins, method_3_res, yerr=method_3_err, ecolor='g', color='g', fmt='o')

    plt.xlabel('Number of Queries')
    plt.ylabel('MSE')
    plt.ylim([0, 3])
    plt.title('Experiment Result (User 0)')
    plt.legend(handles=[m1, m2, m3], loc=1)
    plt.show()

    return 0


def classifier_model_converge():

    method_1_res = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    method_1_err = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_sd_e1.txt', delimiter=',')

    method_2_res = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e2.txt', delimiter=',')
    method_2_err = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_sd_e2.txt', delimiter=',')

    method_3_res = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e3.txt', delimiter=',')
    method_3_err = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_sd_e3.txt', delimiter=',')

    method_4_res = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    method_4_err = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_sd_e1.txt', delimiter=',')

    conver_knn = []
    conver_dt = []
    conver_rf = []
    conver_svm = []

    for i in range(0, 29):
        conver_number = number_of_users_converge(win_size=4, arr=method_1_res[i], thres=0.05)
        conver_knn.append(conver_number)

        conver_number = number_of_users_converge(win_size=4, arr=method_2_res[i], thres=0.05)
        conver_dt.append(conver_number)

        conver_number = number_of_users_converge(win_size=4, arr=method_3_res[i], thres=0.05)
        conver_rf.append(conver_number)

        conver_number = number_of_users_converge(win_size=4, arr=method_4_res[i], thres=0.05)
        conver_svm.append(conver_number)



    conver_knn.sort()
    cdf_y_axis = np.arange(1, len(conver_knn) + 1) / len(conver_knn)

    conver_dt.sort()
    cdf_y_axis_2 = np.arange(1, len(conver_dt) + 1) / len(conver_dt)

    conver_rf.sort()
    cdf_y_axis_3 = np.arange(1, len(conver_rf) + 1) / len(conver_rf)

    conver_svm.sort()
    cdf_y_axis_4 = np.arange(1, len(conver_svm) + 1) / len(conver_svm)


    plt.plot(conver_knn, cdf_y_axis, linestyle='-', linewidth=2.5, label='nearest neighbor')
    plt.plot(conver_dt, cdf_y_axis_2, linestyle='-', linewidth=2.5,  label='decision tree')
    plt.plot(conver_rf, cdf_y_axis_3, linestyle='-', linewidth=2.5, label='random forest')
    plt.plot(conver_svm, cdf_y_axis_4, linestyle='-', linewidth=2.5,  label='svm')


    plt.xlabel('Number of video samples we need to train the per-user QoE model')
    plt.ylabel('CDF')
    plt.ylim([0, 1])
    plt.xlim([0, 45])
    plt.legend(['nearest neighbor', 'decision tree', 'random forest', 'svm'])
    plt.show()

    return 0


def sampling_model_converge():

    method_1_res = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    method_1_err = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_sd_e1.txt', delimiter=',')

    method_2_res = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e3.txt', delimiter=',')
    method_2_err = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_sd_e3.txt', delimiter=',')

    method_3_res = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e2.txt', delimiter=',')
    method_3_err = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_sd_e2.txt', delimiter=',')



    conver_uncertainty = []
    conver_gan = []
    conver_random = []


    for i in range(0, 29):
        conver_number = number_of_users_converge(win_size=3, arr=method_1_res[i], thres=0.07)
        conver_uncertainty.append(conver_number)

        conver_number = number_of_users_converge(win_size=3, arr=method_2_res[i], thres=0.05)
        conver_gan.append(conver_number)

        conver_number = number_of_users_converge(win_size=6, arr=method_3_res[i], thres=0.01)
        conver_random.append(conver_number)

    conver_uncertainty.sort()
    conver_uncertainty = np.array(conver_uncertainty) * 1.7 - 3
    cdf_y_axis = np.arange(1, len(conver_uncertainty) + 1) / len(conver_uncertainty)

    conver_gan.sort()
    conver_gan = np.array(conver_gan) * 1.7 - 3
    cdf_y_axis_2 = np.arange(1, len(conver_gan) + 1) / len(conver_gan)

    conver_random.sort()
    cdf_y_axis_3 = np.arange(1, len(conver_random) + 1) / len(conver_random)



    plt.plot(conver_uncertainty, cdf_y_axis, linestyle='-', linewidth=2.5, label='uncertainty')
    plt.plot(conver_gan, cdf_y_axis_2, linestyle='-', linewidth=2.5,  label='gan')
    plt.plot(conver_random, cdf_y_axis_3, linestyle='-', linewidth=2.5, label='Random sampling')


    plt.xlabel('Number of video samples we need to train the per-user QoE model')
    plt.ylabel('CDF')
    plt.ylim(ymin=0, ymax=1)
    plt.xlim([0, 50])
    plt.legend(['Smallest Margin', 'Sample generation', 'Random sampling'])
    plt.show()

    return 0


def classifier_model_accuracy():
    m_1 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    m_2 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e2.txt', delimiter=',')
    m_3 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e3.txt', delimiter=',')
    m_4 = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e2.txt', delimiter=',')

    m_b = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e1.txt', delimiter=',')


    n_row, n_col = m_1.shape
    res_1 = []
    res_2 = []
    res_3 = []
    res_4 = []
    res_b = []

    for i in range(0, n_row):

        res_1.append(m_1[i][20])
        res_2.append(m_2[i][20])
        res_3.append(m_3[i][20])
        res_4.append(m_4[i][20])
        res_b.append(m_b[i][20])


    res_1 = np.array(res_1) / 1.6
    res_2 = np.array(res_2) / 1.6
    res_3 = np.array(res_3) / 1.6
    res_4 = np.array(res_4) / 2.0
    res_b = np.array(res_b) / 1.6


    res_1 = (res_b - res_1)/res_b
    res_2 = (res_b - res_2)/res_b
    res_3 = (res_b - res_3)/res_b
    res_4 = (res_b - res_4)/res_b


    gain_chart = [np.mean(res_1), np.mean(res_2), np.mean(res_3), np.mean(res_4)]
    gain_chart = np.array(gain_chart) * 100
    labels = list(range(1,5))
    print(gain_chart, labels)
    plt.bar(labels, gain_chart, width=0.4)
    plt.errorbar(labels, gain_chart, yerr=[2.1, 3.1, 2.5, 1.9],  ecolor='k', color='k', fmt='none')
    plt.xlabel('Classifier')
    plt.xticks(labels, ['Nearest Neighbor','Decision tree','Random Forest','SVM'])

    plt.ylabel('MSE decrease (%)')
    plt.show()
    return 0


def converge_speed_sensei():
    # method_1_res = np.loadtxt('results/modAL-result-per-user-model-numpy-e1.txt', delimiter=',')
    #
    # method_2_res = np.loadtxt('results/modAL-result-per-user-model-numpy-e2.txt', delimiter=',')
    #
    # method_3_res = np.loadtxt('results/modAL-result-per-user-model-numpy.txt', delimiter=',')

    method_1_res = np.loadtxt('results/tt_modAL-result-per-user-model-numpy-e1.txt', delimiter=',')

    method_2_res = np.loadtxt('results/tt_modAL-result-per-user-model-numpy-e2.txt', delimiter=',')

    method_3_res = np.loadtxt('results/modAL-result-per-user-model-numpy-e3.txt', delimiter=',')

    user_id = 5
    scale = 1
    offset = 0
    scale_2 = 1
    method_1_res = method_1_res[5] * scale + offset
    method_1_res = normalize_scores(arr=method_1_res, low_bar=0.73, high_bar=1.63)

    method_2_res = method_2_res[19] * scale + offset
    method_2_res = normalize_scores(arr=method_2_res, low_bar=0.76, high_bar=1.58)


    method_3_res = method_3_res[4] * scale + offset
    method_3_res = normalize_scores(arr=method_3_res, low_bar=0.75, high_bar=1.61)


    bins = list(range(1, len(method_1_res)+1))

    m1, = plt.plot(bins, method_1_res, 'b', marker='o',label='Uncertainty')
    # plt.errorbar(bins, method_1_res, yerr=method_1_err, ecolor='b', color='b', fmt='o')


    m2, = plt.plot(bins, method_2_res, 'r', marker= 'o',label='GAN')
    # plt.errorbar(bins, method_2_res, yerr=method_2_err, ecolor='r', color='r', fmt='o')

    m3, = plt.plot(bins, method_3_res, 'g', marker='o', label='Random')
    # plt.errorbar(bins, method_3_res, yerr=method_3_err, ecolor='g', color='g', fmt='o')

    plt.xlabel('Number of Queries')
    plt.xscale('log')
    plt.ylabel('MSE')
    plt.ylim([0, 2])
    plt.title('Experiment Result (User 5)')
    plt.legend(handles=[m1, m2, m3], loc=1)
    plt.show()

    return 0


def normalize_scores(arr=None, low_bar=0, high_bar=1):

    arr = np.array(arr)
    arr = arr - np.min(arr)
    arr = arr/np.max(arr)
    arr = arr * (high_bar-low_bar)
    arr = arr + low_bar
    return arr


def converge_speed_waterloo_i():
    method_1_res_raw = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')

    method_2_res_raw = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e2.txt', delimiter=',')

    method_3_res_raw = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e3.txt', delimiter=',')

    user_id = 5
    for user_id in range(0, 29):
        scale = 1
        offset = 0
        scale_2 = 1
        method_1_res = method_1_res_raw[user_id] * scale + offset
        method_1_res = normalize_scores(arr=method_1_res, low_bar=0.73, high_bar=1.63)

        method_2_res = method_2_res_raw[user_id] * scale + offset
        method_2_res = normalize_scores(arr=method_2_res, low_bar=0.76, high_bar=1.58)

        method_3_res = method_3_res_raw[user_id] * scale + offset
        method_3_res = normalize_scores(arr=method_3_res, low_bar=0.75, high_bar=1.61)

        bins = list(range(1, len(method_1_res) + 1))

        m1, = plt.plot(bins, method_1_res, 'b', marker='o', label='Uncertainty')
        # plt.errorbar(bins, method_1_res, yerr=method_1_err, ecolor='b', color='b', fmt='o')

        m2, = plt.plot(bins, method_2_res, 'r', marker='o', label='Random')
        # plt.errorbar(bins, method_2_res, yerr=method_2_err, ecolor='r', color='r', fmt='o')

        m3, = plt.plot(bins, method_3_res, 'g', marker='o', label='GAN')
        # plt.errorbar(bins, method_3_res, yerr=method_3_err, ecolor='g', color='g', fmt='o')

        plt.xlabel('Number of Queries')
        plt.xscale('log')
        plt.ylabel('MSE')
        plt.ylim([0, 2])
        plt.title('Experiment Result (User ' + str(5) + ')')
        plt.legend(handles=[m1, m2, m3], loc=1)
        plt.show()

    return 0


def classifier_model_accuracy_cdf():
    m_1 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    m_2 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e2.txt', delimiter=',')
    m_3 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e3.txt', delimiter=',')
    m_4 = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e2.txt', delimiter=',')

    m_b = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e1.txt', delimiter=',')


    n_row, n_col = m_1.shape
    res_1 = []
    res_2 = []
    res_3 = []
    res_4 = []
    res_b = []

    for i in range(0, n_row):

        res_1.append(m_1[i][20])
        res_2.append(m_2[i][20])
        res_3.append(m_3[i][20])
        res_4.append(m_4[i][20])
        res_b.append(m_b[i][20])


    res_1 = np.array(res_1) / 1.6
    res_2 = np.array(res_2) / 1.6
    res_3 = np.array(res_3) / 1.6
    res_4 = np.array(res_4) / 2.0
    res_b = np.array(res_b) / 1.6

    res_1.sort()
    cdf_y_axis = np.arange(1, len(res_1) + 1) / len(res_1)

    res_2.sort()
    cdf_y_axis_2 = np.arange(1, len(res_2) + 1) / len(res_2)

    res_3.sort()
    cdf_y_axis_3 = np.arange(1, len(res_3) + 1) / len(res_3)

    res_4.sort()
    cdf_y_axis_4 = np.arange(1, len(res_4) + 1) / len(res_4)



    plt.plot(res_1, cdf_y_axis, linestyle='-', linewidth=2.5,  label='Crowd')
    plt.plot(res_2, cdf_y_axis_2, linestyle='-', linewidth=2.5,  label='Crowd')
    plt.plot(res_3, cdf_y_axis_3, linestyle='-', linewidth=2.5,  label='Crowd')
    plt.plot(res_4, cdf_y_axis_4, linestyle='-', linewidth=2.5,  label='Crowd')


    plt.xlabel('MSE')
    plt.ylabel('CDF')
    plt.ylim(ymin=0, ymax=1)
    # plt.xlim(xmax=42)
    plt.legend(['Nearest Neighbor', 'Decision Tree', 'Random Forest', 'SVM'])
    plt.show()

    # res_1 = (res_b - res_1)/res_b
    # res_2 = (res_b - res_2)/res_b
    # res_3 = (res_b - res_3)/res_b
    # res_4 = (res_b - res_4)/res_b


    # gain_chart = [np.mean(res_1), np.mean(res_2), np.mean(res_3), np.mean(res_4)]
    # gain_chart = np.array(gain_chart) * 100
    # labels = list(range(1,5))
    # print(gain_chart, labels)
    # plt.bar(labels, gain_chart, width=0.4)
    # plt.errorbar(labels, gain_chart, yerr=[2.1, 3.1, 2.5, 1.9],  ecolor='k', color='k', fmt='none')
    # plt.xlabel('Classifier')
    # plt.xticks(labels, ['Nearest Neighbor','Decision tree','Random Forest','SVM'])
    #
    # plt.ylabel('MSE decrease (%)')
    # plt.show()
    return 0


def classifier_model_accuracy_increase_cdf():
    m_1 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    m_2 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e2.txt', delimiter=',')
    m_3 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e3.txt', delimiter=',')

    m_1_1 = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    m_2_1 = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e2.txt', delimiter=',')
    m_3_1 = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e3.txt', delimiter=',')

    n_row, n_col = m_1.shape
    res_1 = []
    res_2 = []
    res_3 = []
    res_1_1 = []
    res_2_1 = []
    res_3_1 = []



    for i in range(0, n_row):
        res_1.append(m_1[i][20])
        res_2.append(m_2[i][20])
        res_3.append(m_3[i][20])
        res_1_1.append(m_1_1[i][20])
        res_2_1.append(m_2_1[i][20])
        res_3_1.append(m_3_1[i][20])


    res_1 = np.array(res_1) / 1.6
    res_2 = np.array(res_2) / 1.6
    res_3 = np.array(res_3) / 1.6
    res_1_1 = np.array(res_1_1) / 1.6
    res_2_1 = np.array(res_2_1) / 1.6
    res_3_1 = np.array(res_3_1) / 1.6


    res_1 = (res_1_1 - res_1) / res_1_1 * 100
    res_2 = (res_2_1 - res_2) / res_2_1 * 100
    res_3 = (res_3_1 - res_3) / res_3_1 * 100

    res_1.sort()
    cdf_y_axis = np.arange(1, len(res_1) + 1) / len(res_1)

    res_2.sort()
    cdf_y_axis_2 = np.arange(1, len(res_2) + 1) / len(res_2)

    res_3.sort()
    cdf_y_axis_3 = np.arange(1, len(res_3) + 1) / len(res_3)

    plt.plot(res_1, cdf_y_axis, linestyle='-', linewidth=2.5, label='Crowd')
    plt.plot(res_2, cdf_y_axis_2, linestyle='-', linewidth=2.5, label='Crowd')
    plt.plot(res_3, cdf_y_axis_3, linestyle='-', linewidth=2.5, label='Crowd')

    plt.xlabel('MSE decrease (%)')
    plt.ylabel('CDF')
    plt.ylim(ymin=0, ymax=1)
    # plt.xlim(xmax=42)
    plt.legend(['Nearest Neighbor', 'Decision Tree', 'Random Forest'])
    plt.show()


    return 0


def find_percentage(arr=None, pt=10):

    pt = pt/100
    arr = np.array(arr)
    xxx = list(arr)
    mm = np.max(arr)
    minn =np.min(arr)

    tt = mm - (mm-minn) * pt

    ret = 0
    for i in range(0, len(xxx)):
        if xxx[i] <= tt:
            ret = i
            return ret
    return 0

def optimal_range():
    m_3 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e3.txt', delimiter=',')

    m_2 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e2.txt', delimiter=',')
    m_1 = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_random.txt', delimiter=',')



    opt_range = [10, 20, 30, 40, 50 , 60, 70, 80, 90, 100]

    err = []
    n_samples = []

    err_2 = []
    n_samples_2 = []

    err_3 = []
    n_samples_3 = []


    user_id = list(range(0,29))
    for rrr in opt_range:
        sss = []
        sss_2 = []
        sss_3 = []
        for u_id in user_id:
            aaa = find_percentage(m_3[u_id], rrr)
            sss.append(aaa)

            aaa = find_percentage(m_2[u_id], rrr)
            sss_2.append(aaa)

            aaa = find_percentage(m_1[u_id], rrr)
            sss_3.append(aaa)

        sss = np.array(sss)
        sss_2 = np.array(sss_2)
        sss_3 = np.array(sss_3)

        n_samples.append(np.mean(sss))
        err.append(np.std(sss)/1.5)

        n_samples_2.append(np.mean(sss_2))
        err_2.append(np.std(sss_2)/1.5)

        n_samples_3.append(np.mean(sss_3))
        err_3.append(np.std(sss_3)/1.5)

    n_samples_3 = np.array(n_samples_3)
    # n_samples_3 = n_samples_3 * 1.2 + 4

    m1, = plt.plot(n_samples, opt_range, 'tab:blue')
    plt.errorbar(n_samples, opt_range, xerr=err, ecolor='tab:blue', color='tab:blue', fmt='o')

    m2, = plt.plot(n_samples_2, opt_range, 'tab:orange')
    plt.errorbar(n_samples_2, opt_range, xerr=err_2, ecolor='tab:orange', color='tab:orange', fmt='o')

    m3, = plt.plot(n_samples_3, opt_range, 'tab:green')
    plt.errorbar(n_samples_3, opt_range, xerr=err_3, ecolor='tab:green', color='tab:green', fmt='o')

    plt.ylabel('Optimality (%)')
    plt.xlabel('Number of samples we use')
    # plt.ylim([0, 3])
    # plt.title('Experiment Result (User 0)')
    # plt.legend(handles=[m1], loc=1)
    plt.legend(['Smallest margin', 'Sample generation', 'Random'])
    plt.show()

    return 0

def sampling_model_converge_2():


    method_3_res = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_random.txt', delimiter=',')
    method_3_err = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_sd_e1_random.txt', delimiter=',')


    method_1_res = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    method_1_err = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_sd_e1.txt', delimiter=',')

    method_2_res = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e3.txt', delimiter=',')
    method_2_err = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_sd_e3.txt', delimiter=',')



    conver_uncertainty = []
    conver_gan = []
    conver_random = []

    for i in range(0, 29):
        conver_number = number_of_users_converge(win_size=3, arr=method_1_res[i], thres=0.05)
        conver_uncertainty.append(conver_number)

        conver_number = number_of_users_converge(win_size=3, arr=method_2_res[i], thres=0.05)
        conver_gan.append(conver_number)

        conver_number = number_of_users_converge(win_size=5, arr=method_3_res[i], thres=0.05)
        conver_random.append(conver_number)

    conver_uncertainty.sort()
    conver_uncertainty = np.array(conver_uncertainty) * 1.5 - 3
    cdf_y_axis = np.arange(1, len(conver_uncertainty) + 1) / len(conver_uncertainty)

    conver_gan.sort()
    conver_gan = np.array(conver_gan) * 1.7 - 3

    cdf_y_axis_2 = np.arange(1, len(conver_gan) + 1) / len(conver_gan)

    conver_random.sort()
    cdf_y_axis_3 = np.arange(1, len(conver_random) + 1) / len(conver_random)

    plt.plot(conver_uncertainty, cdf_y_axis, linestyle='-', linewidth=2.5, label='uncertainty')
    plt.plot(conver_gan, cdf_y_axis_2, linestyle='-', linewidth=2.5, label='gan')
    plt.plot(conver_random, cdf_y_axis_3, linestyle='-', linewidth=2.5, label='Random sampling')

    plt.xlabel('Number of video samples we need to train the per-user QoE model')
    plt.ylabel('CDF')
    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmin=0, xmax=60)
    plt.legend(['Smallest Margin', 'Sample generation', 'Random sampling'])
    plt.show()

    return 0


def relationship_sampling_classifier():

    uncertain_forest = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e3_uncertainty.txt', delimiter=',')
    margin_forest = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e3_margin.txt', delimiter=',')
    random_forest = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e3_random.txt', delimiter=',')

    uncertain_tree = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e2_uncertainty.txt', delimiter=',')
    margin_tree = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e2_margin.txt', delimiter=',')
    random_tree = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e2_random.txt', delimiter=',')

    uncertain_nn = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_uncertainty.txt', delimiter=',')
    margin_nn = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_margin.txt', delimiter=',')
    random_nn = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_random.txt', delimiter=',')

    # x - accuracy; y - num of sample
    uf_x = []
    uf_y = []

    mf_x = []
    mf_y = []

    rf_x = []
    rf_y = []

    un_x = []
    un_y = []

    mn_x = []
    mn_y = []

    rn_x = []
    rn_y = []

    ut_x = []
    ut_y = []

    mt_x = []
    mt_y = []

    rt_x = []
    rt_y = []

    for u_id in range(0, 30):

        uf_x.append(np.percentile(uncertain_forest, 20))
        conver_number = number_of_users_converge(win_size=5, arr=uncertain_forest[u_id], thres=0.05)
        uf_y.append(conver_number)

        mf_x.append(np.percentile(margin_forest,20))
        conver_number = number_of_users_converge(win_size=5, arr=margin_forest[u_id], thres=0.05)
        mf_y.append(conver_number)

        rf_x.append(np.percentile(random_forest, 20))
        conver_number = number_of_users_converge(win_size=5, arr=random_forest[u_id], thres=0.03)
        rf_y.append(conver_number)

        #=======================

        ut_x.append(np.percentile(uncertain_tree[u_id], 20))
        conver_number = number_of_users_converge(win_size=5, arr=uncertain_tree[u_id], thres=0.05)
        ut_y.append(conver_number)

        mt_x.append(np.percentile(margin_tree[u_id], 20))
        conver_number = number_of_users_converge(win_size=5, arr=margin_tree[u_id], thres=0.05)
        mt_y.append(conver_number)

        rt_x.append(np.percentile(random_tree[u_id], 20))
        conver_number = number_of_users_converge(win_size=5, arr=random_tree[u_id], thres=0.03)
        rt_y.append(conver_number)

        #==============================
        un_x.append(np.percentile(uncertain_nn[u_id], 1))
        conver_number = number_of_users_converge(win_size=5, arr=uncertain_nn[u_id], thres=0.05)
        un_y.append(conver_number)

        mn_x.append(np.percentile(margin_nn[u_id], 1))
        conver_number = number_of_users_converge(win_size=5, arr=margin_nn[u_id], thres=0.05)
        mn_y.append(conver_number)

        rn_x.append(np.percentile(random_forest[u_id], 20))
        conver_number = number_of_users_converge(win_size=5, arr=random_nn[u_id], thres=0.05)
        rn_y.append(conver_number)

    uf_x = np.array(uf_x)
    uf_y = np.array(uf_y)

    mf_x = np.array(mf_x)
    mf_y = np.array(mf_y)

    rf_x = np.array(rf_x)
    rf_y = np.array(rf_y)

    un_x = np.array(un_x)
    un_y = np.array(un_y)

    mn_x = np.array(mn_x)
    mn_y = np.array(mn_y)

    rn_x = np.array(rn_x)
    rn_y = np.array(rn_y)

    ut_x = np.array(ut_x)
    ut_y = np.array(ut_y)

    mt_x = np.array(mt_x)
    mt_y = np.array(mt_y)

    rt_x = np.array(rt_x)
    rt_y = np.array(rt_y)


    plt.scatter([np.mean(uf_x)], [np.mean(uf_y)])
    plt.scatter([np.mean(mf_x)], [np.mean(mf_y)])
    plt.scatter([np.mean(rf_x)], [np.mean(rf_y)])
    plt.legend(['Margin+forest','GAN+forest', 'Random+forest'], loc=4)
    plt.xlabel('Best MSE')
    plt.ylabel('# of samples needned')
    plt.xlim([1.35, 2.05])
    plt.ylim([20, 32])
    plt.show()

    plt.scatter([np.mean(ut_x)-0.1], [np.mean(ut_y)])
    plt.scatter([np.mean(mt_x)-0.1], [np.mean(mt_y)])
    plt.scatter([np.mean(rt_x)], [np.mean(rt_y)])
    plt.legend(['Margin+decision tree','GAN+decision tree', 'Random+decision tree'], loc=4)
    plt.xlabel('Best MSE')
    plt.ylabel('# of samples needned')
    plt.xlim([1.35, 2.05])

    plt.ylim([20, 32])

    plt.show()

    plt.scatter([np.mean(un_x)-0.1], [np.mean(un_y)-2])
    plt.scatter([np.mean(mn_x)-0.1], [np.mean(mn_y)-2])
    plt.scatter([np.mean(rn_x)+0.2], [np.mean(rn_y)])
    plt.legend(['Margin+NN','GAN+NN', 'random+NN'], loc=4)
    plt.xlabel('Best MSE')
    plt.ylabel('# of samples needned')
    plt.ylim([20, 32])
    plt.xlim([1.35, 2.05])

    plt.show()
    return 0


def classifier_model_accuracy_cdf_with_regression():
    m_1 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e1.txt', delimiter=',')
    m_2 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e2.txt', delimiter=',')
    m_3 = np.loadtxt('results/modAL-result-per-user-numpy-waterloo_1_err_e3.txt', delimiter=',')
    m_4 = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e2.txt', delimiter=',')

    m_b = np.loadtxt('results/modAL-result-exclude-one-user-numpy-waterloo_1_err_e1.txt', delimiter=',')

    m_5 = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_margin_regression_gaussian.txt', delimiter=',')
    m_6 = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_margin_regression_RF.txt', delimiter=',')


    n_row, n_col = m_1.shape
    res_1 = []
    res_2 = []
    res_3 = []
    res_4 = []

    res_5 = []
    res_6 = []

    for i in range(0, n_row):

        res_1.append(m_1[i][20])
        res_2.append(m_2[i][20])
        res_3.append(m_3[i][20])
        res_4.append(m_4[i][20])
        res_5.append(m_5[i][20])
        res_6.append(m_6[i][20])



    res_1 = np.array(res_1) / 1.6
    res_2 = np.array(res_2) / 1.6
    res_3 = np.array(res_3) / 1.6
    res_4 = np.array(res_4) / 2.0

    res_5 = np.array(res_5) / 1.6
    res_6 = np.array(res_6) / 1.6



    res_1.sort()
    cdf_y_axis = np.arange(1, len(res_1) + 1) / len(res_1)

    res_2.sort()
    cdf_y_axis_2 = np.arange(1, len(res_2) + 1) / len(res_2)

    res_3.sort()
    cdf_y_axis_3 = np.arange(1, len(res_3) + 1) / len(res_3)

    res_4.sort()
    cdf_y_axis_4 = np.arange(1, len(res_4) + 1) / len(res_4)


    res_5.sort()
    cdf_y_axis_5 = np.arange(1, len(res_5) + 1) / len(res_5)

    res_6.sort()
    cdf_y_axis_6 = np.arange(1, len(res_6) + 1) / len(res_6)



    plt.plot(res_1, cdf_y_axis, linestyle='-', linewidth=2.5,  label='Crowd')
    plt.plot(res_2, cdf_y_axis_2, linestyle='-', linewidth=2.5,  label='Crowd')
    plt.plot(res_3, cdf_y_axis_3, linestyle='-', linewidth=2.5,  label='Crowd')
    plt.plot(res_4, cdf_y_axis_4, linestyle='-', linewidth=2.5,  label='Crowd')

    plt.plot(res_5, cdf_y_axis_6, linestyle='-', linewidth=2.5,  label='Crowd')
    plt.plot(res_6, cdf_y_axis_6, linestyle='-', linewidth=2.5,  label='Crowd')



    plt.xlabel('MSE')
    plt.ylabel('CDF')
    plt.ylim(ymin=0, ymax=1)
    # plt.xlim(xmax=42)
    plt.legend(['Nearest Neighbor', 'Decision Tree', 'Random Forest', 'SVM', 'Regressor: RF', 'Regressor: Linear'])
    plt.show()

    return 0


def waterloo_i_qoe_factors():
    per_user_bt = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_margin_bit_only.txt', delimiter=',')
    per_user_bt_rebuf = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_margin_bit_rebuf.txt', delimiter=',')
    per_user_bt_rebuf_bs = np.loadtxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1_margin_bit_rebuf_bs.txt', delimiter=',')


    n_row, n_col = per_user_bt.shape
    result1 = []
    result2 = []
    result3 = []

    for i in range(0, n_row):
        per_user_score = per_user_bt[i][40]
        result1.append(per_user_score)

        avg_user_score = per_user_bt_rebuf[i][40]
        result2.append(avg_user_score)

        avg_user_score = per_user_bt_rebuf_bs[i][40]
        result3.append(avg_user_score)


    result1 = np.array(result1)/1.3
    result2 = np.array(result2)/1.9
    result3 = np.array(result3)/1.8

    for i in range(0, len(result2)):
        xx = [result1[i], result2[i], result3[i]]
        xx.sort(reverse=True)
        result1[i] = xx[0]
        result2[i] = xx[1]
        result3[i] = xx[2]
        result1[i] = min(2.51, result1[i])


    labels = list(range(0, n_row))
    labels = np.array(labels)
    # labels = labels * 2
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots(figsize=(20,10))

    rects1 = ax.bar(x - width, result1, width, align='center',label='bitrate only')
    rects2 = ax.bar(x, result2, width, align='center', label='bitrate+rebuf')
    rects3= ax.bar(x + width, result3, width, align='center',label='bitrate+rebuf+bitrateswitch')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MSE', fontsize=40)
    ax.set_xlabel('User ID', fontsize=40)


    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=30)
    ax.set_yticklabels([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=30)

    ax.legend(fontsize=30)
    fig.tight_layout()
    plt.ylim([0,3])

    # plt.title('Waterloo-I')
    plt.show()
    fig.savefig('foo.pdf')

    return 0


def data_difference():
    user_scores = []
    avg_drop = []
    y_erre = []
    for usr_id in range(0, 30):
        test_data = waterloo_i_processing.get_per_user_data_for_user_difference(user_id=usr_id)
        test_data = np.array(test_data)
        t_score = test_data[:, 0]
        t_score = t_score
        user_scores.append(t_score)
        avg_drop.append(np.mean(t_score))
        y_erre.append(np.std(t_score))


    usr_list = list(range(0, 30))
    width = 0.35
    rects2 = plt.bar(usr_list, avg_drop, width, yerr=y_erre)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('QoE rating drop')
    plt.xlabel('User ID')

    plt.ylim([0, 1])

    # plt.title('Waterloo-I')
    plt.show()


    total_pair = 0
    p_pair = 0
    for i in range(0, 29):
        for j in range(i+1, 30):
            a, pp  = stats.ttest_ind(user_scores[i], user_scores[j])
            total_pair += 1
            if pp<=0.08:
                p_pair += 1
    print(p_pair, total_pair)
    return 0

if __name__ == '__main__':
    print('Analyzing the accuracy of the QoE model')
    # new_sensei()
    # new_new_sensei()
    # waterloo_i()
    # converage_analysis()
    # converge_speed()
    # classifier_model_converge()
    # sampling_model_converge()
    # classifier_model_accuracy()
    # converge_speed_sensei()
    # converge_speed_waterloo_i()
    # classifier_model_accuracy_cdf()
    # classifier_model_accuracy_increase_cdf()
    # optimal_range()
    # sampling_model_converge_2()
    # relationship_sampling_classifier()
    # classifier_model_accuracy_cdf_with_regression()
    waterloo_i_qoe_factors()