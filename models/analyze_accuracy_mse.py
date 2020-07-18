import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def sys_main():

    folder_path = 'results/'

    final_results = []
    for usr in range(0, 1):
        # e1: uncertainty; e2: random; e3: alce
        file_path_e1 = folder_path + 'exclude_user_' + str(usr) + '_E3_class_10.txt'
        file_path_e2 = folder_path + 'hdtv_user_' + str(usr) + '_E3_class_10.txt'


        e1_data = np.loadtxt(file_path_e1, delimiter=',') * 200
        e2_data = np.loadtxt(file_path_e2, delimiter=',') * 200

        bins = list(range(0, 350, 20))

        e1_sample_data = e1_data[:, bins]
        e1_sample_avg = np.mean(e1_sample_data, axis=0)
        e1_sample_std_error = np.std(e1_sample_data, axis=0)/np.sqrt(e1_sample_data.shape[0])

        e2_sample_data = e2_data[:, bins]
        e2_sample_avg = np.mean(e2_sample_data, axis=0)
        e2_sample_std_error = np.std(e2_sample_data, axis=0)/np.sqrt(e2_sample_data.shape[0])

        bins = np.array(bins) + 10
        uncert, = plt.plot(bins, e1_sample_avg, 'b', label='Model for user ' + str(usr))
        plt.errorbar(bins, e1_sample_avg, yerr=e1_sample_std_error, ecolor='b', color='b', fmt='o')

        rd, = plt.plot(bins, e2_sample_avg, 'r', label='Model trained by other user scores')
        plt.errorbar(bins, e2_sample_avg, yerr=e2_sample_std_error, ecolor='r', color='r', fmt='o')


        plt.xlabel('Number of Queries')
        plt.ylabel('MSE')
        plt.ylim([0, 2500])
        plt.title('Experiment Result (User ' + str(usr)+')')
        plt.legend(handles=[uncert, rd], loc=1)
        plt.show()

        print(e1_sample_avg)

        print(e2_sample_avg)
    return 0


if __name__ == '__main__':
    print('Analyzing the accuracy of the QoE model')
    sys_main()
