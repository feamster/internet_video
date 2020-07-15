import copy
import matplotlib.pyplot as plt
import numpy as np

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


def run(trn_ds, tst_ds, lbr, model, qs, quota, cost_matrix):
    C_in, C_out = [], []

    for i in range(quota + 1):
        # Standard usage of libact objects
        if i > 0:
            ask_id = qs.make_query()
            lb = lbr.label(trn_ds.data[ask_id][0])
            trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        # print('--------------')
        # print(trn_ds.get_labeled_entries()[0])
        # print('--------------')
        # print(trn_ds.get_labeled_entries()[1])

        trn_X = trn_ds.get_labeled_entries()[0]
        trn_y = trn_ds.get_labeled_entries()[1]

        tst_X = tst_ds.get_labeled_entries()[0]
        tst_y = tst_ds.get_labeled_entries()[1]

        # trn_X, trn_y = zip(*trn_ds.get_labeled_entries())
        # tst_X, tst_y = zip(*tst_ds.get_labeled_entries())
        C_in = np.append(C_in,
                         calc_cost(trn_y, model.predict(trn_X), cost_matrix))
        C_out = np.append(C_out,
                          calc_cost(tst_y, model.predict(tst_X), cost_matrix))

    return C_in, C_out


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


def split_train_test(X, y, test_size, n_class):

    target = np.unique(y)
    # mapping the targets to 0 to n_classes-1
    # y = np.array([np.where(target == i)[0][0] for i in data['target']])

    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=test_size, stratify=y)

    # making sure each class appears ones initially
    init_y_ind = np.array([np.where(y_trn == i)[0][0] for i in range(len(target))])

    y_ind = np.array([i for i in range(len(X_trn)) if i not in init_y_ind])
    trn_ds = Dataset(
        np.vstack((X_trn[init_y_ind], X_trn[y_ind])),
        np.concatenate((y_trn[init_y_ind], [None] * (len(y_ind)))))

    tst_ds = Dataset(X_tst, y_tst)

    fully_labeled_trn_ds = Dataset(
        np.vstack((X_trn[init_y_ind], X_trn[y_ind])),
        np.concatenate((y_trn[init_y_ind], y_trn[y_ind])))

    cost_matrix = 1.0 * np.ones([len(target), len(target)])
    for ii in range(0, len(target)):
        for jj in range(0, len(target)):
            cost_matrix[ii, jj] = abs(ii - jj) / n_class

    np.fill_diagonal(cost_matrix, 0)

    return trn_ds, tst_ds, fully_labeled_trn_ds, cost_matrix


def save_file(file_name, data):
    np.savetxt(file_name, data, delimiter=',')
    return 0


def train_for_user(user_id=1, device_type='uhdtv', n_class=10):
    test_data = waterloo_iv_processing.get_per_user_data(user_id=user_id, device=device_type,
                                                         video_name=['sports', 'document', 'nature', 'game', 'movie'])
    X, y = processing_training_data(n_class=n_class, train_data=test_data)
    test_size = 0.2  # the percentage of samples in the dataset that will be
    quota = 350  # number of samples to query

    result = {'E1': [], 'E2': [], 'E3': []}
    for i in range(20):
        print('exp:', i)
        trn_ds, tst_ds, fully_labeled_trn_ds, cost_matrix = split_train_test(X=X, y=y, test_size=test_size, n_class=n_class)
        trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        model = SVM(kernel='rbf', decision_function_shape='ovr')

        qs = UncertaintySampling(
            trn_ds, method='sm', model=SVM(decision_function_shape='ovr'))
        _, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota, cost_matrix)
        result['E1'].append(E_out_1)

        qs2 = RandomSampling(trn_ds2)
        _, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota, cost_matrix)
        result['E2'].append(E_out_2)

        qs3 = ALCE(trn_ds3, cost_matrix, SVR())
        _, E_out_3 = run(trn_ds3, tst_ds, lbr, model, qs3, quota, cost_matrix)
        result['E3'].append(E_out_3)

    E_out_1 = np.mean(result['E1'], axis=0)
    E_out_2 = np.mean(result['E2'], axis=0)
    E_out_3 = np.mean(result['E3'], axis=0)

    save_file('results/'+device_type+'_user_'+str(user_id)+'_E1_class_'+str(n_class)+'.txt', result['E1'])
    save_file('results/'+device_type+'_user_'+str(user_id)+'_E2_class_'+str(n_class)+'.txt', result['E2'])
    save_file('results/'+device_type+'_user_'+str(user_id)+'_E3_class_'+str(n_class)+'.txt', result['E3'])

    print("Uncertainty: ", E_out_1[::5].tolist())
    print("Random: ", E_out_2[::5].tolist())
    print("ALCE: ", E_out_3[::5].tolist())

    query_num = np.arange(0, quota + 1)
    uncert, = plt.plot(query_num, E_out_1, 'g', label='Uncertainty sampling')
    rd, = plt.plot(query_num, E_out_2, 'k', label='Random')
    alce, = plt.plot(query_num, E_out_3, 'r', label='ALCE')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(handles=[uncert, rd, alce], loc=3)
    plt.show()


def sys_main():

    usr_list = list(range(0, 29))

    for usr in usr_list:
        print('User:', usr)
        train_for_user(user_id=usr, device_type='phone', n_class=10)
    return 0


if __name__ == '__main__':
    sys_main()
