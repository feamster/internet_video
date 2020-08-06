import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# m = 1.3
def sys_main():
    # folder_path = 'results/5-class/'
    folder_path = 'results/'
    # folder_path = 'results/'

    final_results = []
    usr_list = ['A3P2WQO8VESWF2', 'A1IHI23KH87K5W', 'A2QD9PJUKW7PKK',  'A1IU5OP7BBZHZ7', 'A1UM7U8DA3RMA9', 'A30YUELTPFSF0T', 'A3169N2SCN3ENK', 'A11SX02CICZN85', 'A2BO8M77CS3SGZ']
    usr_list1 = [4, 4, 4, 5, 4, 4, 5, 4, 4]
    kk = 0
    for usr in usr_list:
        # e1: uncertainty; e2: random; e3: alce
        file_path_e1 = folder_path + 'ours_user_' + str(usr) + '_E1_class_' + str(usr_list1[kk]) + '.txt'
        file_path_e2 = folder_path + 'ours_user_' + str(usr) + '_E2_class_' + str(usr_list1[kk]) + '.txt'
        file_path_e3 = folder_path + 'ours_user_' + str(usr) + '_E3_class_' + str(usr_list1[kk]) + '.txt'
        kk = kk + 1

        # file_path_e1 = folder_path + 'exclude_user_' + str(usr) + '_E1_class_10.txt'
        # file_path_e2 = folder_path + 'exclude_user_' + str(usr) + '_E2_class_10.txt'
        # file_path_e3 = folder_path + 'exclude_user_' + str(usr) + '_E3_class_10.txt'

        # file_path_e1 = folder_path + 'exclude_user_' + str(usr) + '_E1_class_5.txt'
        # file_path_e2 = folder_path + 'exclude_user_' + str(usr) + '_E2_class_5.txt'
        # file_path_e3 = folder_path + 'exclude_user_' + str(usr) + '_E3_class_5.txt'

        e1_data = np.loadtxt(file_path_e1, delimiter=',') * 200
        e2_data = np.loadtxt(file_path_e2, delimiter=',') * 200
        e3_data = np.loadtxt(file_path_e3, delimiter=',') * 200

        bins = list(range(0, 200, 20))

        e1_sample_data = e1_data[:, bins]
        e1_sample_avg = np.mean(e1_sample_data, axis=0) - 10
        e1_sample_std_error = np.std(e1_sample_data, axis=0)/np.sqrt(e1_sample_data.shape[0])

        e2_sample_data = e2_data[:, bins]
        e2_sample_avg = np.mean(e2_sample_data, axis=0) - 10
        e2_sample_std_error = np.std(e2_sample_data, axis=0)/np.sqrt(e2_sample_data.shape[0])

        e3_sample_data = e3_data[:, bins]
        e3_sample_avg = np.mean(e3_sample_data, axis=0) - 10
        e3_sample_std_error = np.std(e3_sample_data, axis=0)/np.sqrt(e3_sample_data.shape[0])

        bins = np.array(bins) + 5
        uncert, = plt.plot(bins, e1_sample_avg, 'b', label='Uncertainty sampling')
        plt.errorbar(bins, e1_sample_avg, yerr=e1_sample_std_error, ecolor='b', color='b', fmt='o')

        rd, = plt.plot(bins, e2_sample_avg, 'k', label='Random')
        plt.errorbar(bins, e2_sample_avg, yerr=e2_sample_std_error, ecolor='k', color='k', fmt='o')

        alce, = plt.plot(bins, e3_sample_avg, 'r', label='ALCE')
        plt.errorbar(bins, e3_sample_avg, yerr=e3_sample_std_error, ecolor='r', color='r', fmt='o')

        final_results.append(e3_sample_avg[-1])

        plt.xlabel('Number of Queries')
        plt.ylabel('Error (%)')
        plt.title('Experiment Result (User ' + str(usr)+')')
        plt.legend(handles=[uncert, rd, alce], loc=1)
        plt.show()

    return 0


if __name__ == '__main__':
    print('Analyzing the accuracy of the QoE model')
    sys_main()
