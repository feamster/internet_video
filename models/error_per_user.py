import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def sys_main():
    result1 = [11.059829059829058, 13.952991452991455, 13.717948717948719, 12.863247863247866, 13.2008547008547,
               9.794871794871794, 12.222222222222221, 22.551282051282055, 14.012820512820515, 15.547008547008545,
               16.273504273504273, 15.423076923076923, 15.833333333333334, 11.80769230769231, 10.948717948717949,
               13.26068376068376, 19.824786324786324, 12.837606837606838, 16.051282051282048, 14.33760683760684,
               12.747863247863245, 12.401709401709402, 15.585470085470087, 14.52136752136752, 10.786324786324785,
               16.585470085470085, 10.14102564102564, 16.5]
    result2 = [35.31111111111112, 34.1888888888889, 30.544444444444444, 30.577777777777776, 36.77777777777778,
               28.877777777777776, 34.488888888888894, 69.42222222222223, 35.73333333333333, 34.022222222222226,
               39.42222222222223, 37.711111111111116, 35.58888888888889, 39.16666666666667, 31.522222222222222,
               34.88888888888889, 39.23333333333334, 30.9, 35.211111111111116, 31.377777777777784, 33.88888888888889,
               37.55555555555556, 31.600000000000005, 34.08888888888889, 30.322222222222223, 66.55555555555554,
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


if __name__ == '__main__':
    print('Analyzing the accuracy of the QoE model')
    sys_main()