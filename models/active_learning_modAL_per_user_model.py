import copy
import matplotlib.pyplot as plt
import numpy as np
import random

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
import waterloo_iv_processing

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


def processing_training_data(n_class=10, train_data=None):
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
        error = np.sum(Y_dis)
    if err_cal == 'Err':
        # error = calc_cost(Y_truth, Y_predict, cost_matrix)
        Y_dis = Y_predict - Y_truth
        Y_dis = np.absolute(Y_dis)
        error = np.mean(Y_dis)

    return error

def get_init_train(X_trn_all, y_trn_all, n_of_class=10):
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
        print('exp:', i)
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

        # print('query no.', 0, file=f_2)
        # print('predictions:', predictions_0, file=f_2)
        # print('MSE:', err_0, file=f_2)

        for j in range(n_queries):
            query_index, query_instance = learner.query(X_trn)
            X_qry, y_qry = X_trn[query_index].reshape(1, -1), y_trn[query_index].reshape(1, )
            learner.teach(X=X_qry, y=y_qry)
            X_trn, y_trn = np.delete(X_trn, query_index, axis=0), np.delete(y_trn, query_index)
            predictions = learner.predict(X_tst)
            err = error_calculation(predictions, y_tst)
            # print('query no.', j+1, file=f_2)
            # print('predictions:', predictions, file=f_2)
            # print('MSE:', err, file=f_2)
            performance_history[j].append(err)

    avg_err = []
    for i in range(n_queries):
        avg_err.append(np.mean(performance_history[i]))

    return avg_err


def train_for_user(fd, user_id=1, device_type='uhdtv', n_class=10):
    test_data = waterloo_iv_processing.get_per_user_data(user_id=user_id, device=device_type,
                                                         video_name=['sports', 'document', 'nature', 'game', 'movie'])

    X, y = processing_training_data(n_class=n_class, train_data=test_data)
    
    test_size = 0.2  # the percentage of samples in the dataset that will be
    rep_times = 10
    n_queries = 350

    err = []

    for name, clf in zip(names, classifiers):
        print('model:', name)
        print('model:', name, file=fd)
        E = run_model(X, y, test_size, rep_times, n_queries, clf, fd)
        err.append(E[-1])
        # print(E[-1], file=f_3)
        print(E, file=fd)

    return err

def sys_main():

    usr_list = list(range(0, 29))
    # usr_list = list(range(0, 1))
    fd = open('results/modAL-result-per-user-model.txt','w')
    E1 = []
    E2 = []
    E3 = []
    for usr in usr_list:
        print('User:', usr)
        print('User:', usr, file=fd)
        err = train_for_user(fd, user_id=usr, device_type='hdtv', n_class=10)
        E1.append(err[0])
        E2.append(err[1])
        E3.append(err[2])
    print('\n', file=fd)
    print(E1, file=fd) # nearest neighbor
    print(E2, file=fd) # decision tree
    print(E3, file=fd) # random forest
    return 0

if __name__ == '__main__':
    sys_main()
