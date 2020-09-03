import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR

# libact classes
from libact.base.dataset import Dataset
from libact.models import SVM
from libact.query_strategies.multiclass import ActiveLearningWithCostEmbedding as ALCE
from libact.query_strategies import UncertaintySampling, RandomSampling
from libact.labelers import IdealLabeler
from libact.utils import calc_cost

# private class
import waterloo_i_processing

# modAL
from modAL.models import ActiveLearner

# Classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# names = ["Nearest Neighbors"]
# classifiers = [KNeighborsClassifier(3)]

names = ["Nearest Neighbors", "Decision Tree", "Random Forest"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5), 
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    ]


def process_label_data(n_class, label_data):
    Y = []
    unit_class = 1.0 / n_class
    for i in range(0, len(label_data)):
        this_class = int(label_data[i] / unit_class)
        if this_class >= n_class:
            this_class = n_class - 1
        Y.append(this_class)
    Y = np.array(Y)
    return Y


def processing_training_data(n_class=5, train_data=None):
    '''
    :param n_class:
    :param train_data:
    :return: X features; Y labels
    '''
    if train_data is None:
        print('No training data')
        return None
    train_data = np.array(train_data)
    row, col = train_data.shape
    X = train_data[:, 1:col]
    Y = process_label_data(n_class=n_class, label_data=train_data[:, 0])
    return X, Y


def error_calculation(Y_truth=None, Y_predict=None,  err_cal='MSE'):
    if Y_predict is None or Y_truth is None:
        print('No input')
        return 0
    error = 0.0
    if err_cal == 'MSE':
        Y_dis = Y_predict - Y_truth
        Y_dis = np.power(Y_dis, 2)
        error = np.mean(Y_dis)
    if err_cal == 'Err':
        # error = calc_cost(Y_truth, Y_predict, cost_matrix)
        Y_dis = Y_predict - Y_truth
        Y_dis = np.absolute(Y_dis)
        error = np.mean(Y_dis)

    return error


def get_init_train(X_trn_all, y_trn_all, n_of_class=5):
    trn = []
    for j in range(len(y_trn_all)):
        trn.append((y_trn_all[j], j))
    random.shuffle(trn)
    y_val_occur = [False for i in range(n_of_class)]
    X_trn_min, y_trn_min = [], []
    X_trn, y_trn = [], []
    for y_val, idx in trn:
        if y_val_occur[y_val] == False:
            X_trn_min.append(X_trn_all[idx])
            y_trn_min.append(y_trn_all[idx])
            y_val_occur[y_val] = True
        else:
            X_trn.append(X_trn_all[idx])
            y_trn.append(y_trn_all[idx])
    X_trn_min = np.array(X_trn_min)
    y_trn_min = np.array(y_trn_min)
    X_trn = np.array(X_trn)
    y_trn = np.array(y_trn)
    return X_trn_min, y_trn_min, X_trn, y_trn


def run_model(X, y, test_size, rep_times, n_queries, estimator, fd):
    performance_history = [[] for i in range(n_queries)]
    for i in range(rep_times):
        # print('exp:', i)
        # print('exp:', i, file=fd)
        
        n_labled_examples = X.shape[0]
        X_trn_all, X_tst, y_trn_all, y_tst = train_test_split(X, y, test_size=test_size, stratify=y)
        # get initial training set, which size = n_class
        X_trn_min, y_trn_min, X_trn, y_trn = get_init_train(X_trn_all, y_trn_all)
        # print('ground truth:', y_tst, file=f_2)

        learner = ActiveLearner(estimator=estimator, X_training=X_trn_min, y_training=y_trn_min)

        # prediction with no query
        predictions_0 = learner.predict(X_tst)
        err_0 = error_calculation(predictions_0, y_tst)

        for j in range(n_queries):
            query_index, query_instance = learner.query(X_trn)
            X_qry, y_qry = X_trn[query_index].reshape(1, -1), y_trn[query_index].reshape(1, )
            learner.teach(X=X_qry, y=y_qry)
            X_trn, y_trn = np.delete(X_trn, query_index, axis=0), np.delete(y_trn, query_index)
            predictions = learner.predict(X_tst)
            err = error_calculation(predictions, y_tst)
            performance_history[j].append(err)

    avg_err = []
    sd = []
    for i in range(n_queries):
        avg_err.append(np.mean(performance_history[i]))
        sd.append(np.std(performance_history[i])/np.sqrt(rep_times))

    return avg_err, sd


def train_for_user(fd, user_id=1, n_class=5, data_id=None):
    if data_id is None:
        test_data = waterloo_i_processing.get_per_user_data(user_id=user_id)
    else:
        test_data = waterloo_i_processing.get_per_user_data(user_id=data_id)
    X, y = processing_training_data(n_class=n_class, train_data=test_data)

    test_size = 0.2  # the percentage of samples in the dataset that will be
    rep_times = 50
    n_queries = 120

    err = []
    all_sd = []

    for name, clf in zip(names, classifiers):
        # print('model:', name)
        # print('model:', name, file=fd)
        if name == names[0]:
            E, sd = run_model(X, y, test_size, rep_times, n_queries, clf, fd)
            err = E
            all_sd = sd

        # print(E[-1], file=f_3)

        # print(E, file=fd)
        # print(sd, file=fd)
        # for x in E: print(x, file=fd)

    return err, all_sd


def sys_main():
    usr_list = list(range(0, 30))

    fd = open('results/modAL-result-per-user-model-1-10.txt','w')
    E1 = []
    E2 = []
    E3 = []
    sd1 = []
    sd2 = []
    sd3 = []
    # for usr in usr_list:
    for i in range(len(usr_list)):
        E1_all = []
        E2_all = []
        E3_all = []
        # print('User:', usr_list[i], usr_list[i])
        # print('User:', usr_list[i], usr_list[i], file=fd)
        err, sd = train_for_user(fd, user_id=usr_list[i], n_class=5, data_id=usr_list[i])
        # E1_all.append(err[0])
        # E2_all.append(err[1])
        # E3_all.append(err[2])
        E1.append(err)
        sd1.append(sd)
        print(err[-1])
        print(sd1[-1])
        # sd1.append(sd[0])
        # sd2.append(sd[1])
        # sd3.append(sd[2])


        # E1.append(sum(E1_all)/len(E1_all))
        # E2.append(sum(E2_all)/len(E2_all))
        # E3.append(sum(E3_all)/len(E3_all))
        # break

    E1 = np.array(E1)
    sd1 = np.array(sd1)

    np.savetxt('results/tt_modAL-result-per-user-numpy-waterloo_1_err_e1.txt', E1, delimiter=',')
    np.savetxt('results/tt_modAL-result-per-user-numpy-waterloo_1_sd_e1.txt', sd1, delimiter=',')

    # print(np.mean(E1),np.mean(E2),np.mean(E3))

    # print('\n', file=fd)
    # for x in E1: print(x, file=fd)
    # print('\n', file=fd)
    # for x in sd1: print(x, file=fd)
    # print('\n', file=fd)
    # for x in E2: print(x, file=fd)
    # print('\n', file=fd)
    # for x in sd1: print(x, file=fd)
    # print('\n', file=fd)
    # for x in E3: print(x, file=fd)
    # print('\n', file=fd)
    # for x in sd1: print(x, file=fd)


    # print(E1, file=fd) # nearest neighbor
    # print(E2, file=fd) # decision tree
    # print(E3, file=fd) # random forest
    return 0

if __name__ == '__main__':
    sys_main()
