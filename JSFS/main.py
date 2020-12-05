# pylint: disable = no-member
# --- coding: utf-8 ---
import json

import numpy as np
import numpy.matlib
import pandas as pd
import scipy.io as scio
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, auc, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading

with open("./config.json", 'r') as load_file:
    CONFIG = json.load(load_file)
ck_name_field = ['ant1', 'ivy2', 'jedit4',
                 'lucene2', 'synapse1', 'velocity1', 'xalan2']
nasa_name_field = ['cm1', 'kc3', 'mc2', 'mw1', 'pc1', 'pc3', 'pc4', 'pc5']


def JSFS(X: np.ndarray, y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray, name: str):
    # reference: Jiang, Bingbing, et al.
    # "Joint semi-supervised feature selection and classification through Bayesian approach."
    # Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. 2019.
    print('========== JSFS ==========')
    # --- Input & Initialize---
    # np.set_printoptions(threshold=np.inf)
    n = len(X)
    d = len(X[0])
    y.resize((n, 1))
    testSize = len(test_X)
    # labeled sample ratio
    # labelRatio = 0.5
    labelRatio = CONFIG[name]['labelRatio']
    l = int(n * labelRatio)
    u = n - l
    # γ and µ are super parameters
    # Gamma = 0.001
    Gamma = CONFIG[name]['Gamma']
    # Mu = 0.9
    Mu = CONFIG[name]['Mu']
    # Beta = 0.005
    Beta = 5
    Omega = np.zeros((d, 1))
    Omega[:] = 0.5
    Lambda_vector = np.zeros((u, 1))
    Lambda_vector[:] = 0.5
    A = np.zeros((d, d))
    for i in range(d):
        A[i, i] = 0.001
    C = np.zeros((u, u))
    for i in range(u):
        C[i, i] = 0.001
    # --- Construct the affinity matrix S and graph Laplacian L via KNN ---
    print('Construct the affinity matrix S and graph Laplacian L via KNN')
    trainData_X = X
    trainData_Y = y.ravel()     # y and trainData_Y address the same memory
    # replace the original -1 label with 0, because in this method -1 means no label
    for i in range(n):
        trainData_Y[i] = 0 if trainData_Y[i] == -1 else trainData_Y[i]
    trainData_Y[l:] = -1
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(trainData_X[:l], trainData_Y[:l])
    S = np.zeros((n, n))
    D = np.zeros((n, n))
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if trainData_Y[i] == trainData_Y[j] and trainData_Y[i] != -1:
                S[i][j] = 10
            elif (trainData_Y[i] == -1 and trainData_Y[j] > -1) and (KNN.predict(trainData_X[i:i+1]) == trainData_Y[j]):
                S[i][j] = 1
            elif (trainData_Y[j] == -1 and trainData_Y[i] > -1) and (KNN.predict(trainData_X[j:j+1]) == trainData_Y[i]):
                S[i][j] = 1
            else:
                S[i][j] = 0
            S[j][i] = S[i][j]
        D[i, i] = sum(S[i, :])
        percent = 100 * (float((2*n-i)*(i+1)) / ((n+1)*n))
        show_str = ('[%%-%ds]' % 50) % (int(50*percent/100) * "#")
        print('\r%s %d%%' % (show_str, percent), end='')
    L = D - S
    # --- Obtain the pseudo laber vector y_u via label progation ---
    print('\nObtain the pseudo laber vector y_u via label progation')
    LGC_rbf = LabelSpreading(kernel='knn', gamma=20,
                             n_neighbors=7, max_iter=150)
    LGC_rbf.fit(trainData_X, trainData_Y)
    trainData_Y[l:] = LGC_rbf.predict(trainData_X[l:])
    # change 0 back to the -1
    """ for i in range(n):
        trainData_Y[i] = -1 if trainData_Y[i] == 0 else trainData_Y[i] """
    # --- Data preprocessing - Normalized for X, y ---
    # min_max_scaler = preprocessing.MinMaxScaler((0, 0.0001))
    min_max_scaler = preprocessing.MinMaxScaler(
        (0, CONFIG[name]['xMaxScaler']))
    X = min_max_scaler.fit_transform(X)
    test_X = min_max_scaler.transform(test_X)
    # --- Convergence ---
    B = Gamma * np.dot(np.dot(X.T, L), X)
    Lambda = np.matlib.identity(n)
    Sigma = np.zeros((n, 1))
    E = np.zeros((n, n))
    P = np.zeros((u, u))
    k_lambda = np.zeros((u, 1))
    Eu = np.zeros((u, u))
    O = np.zeros((u, u))
    Omega_old = np.ones((d, 1))
    Lambda_vector_old = np.zeros((u, 1))
    g_omega = np.zeros((d, 1))
    H_omega = np.zeros((d, d))
    Sig_omega = np.zeros((d, d))
    g_lambda = np.zeros((u, 1))
    H_lambda = np.zeros((u, u))
    Sig_lambda = np.zeros((u, u))
    G = np.zeros((d, d))
    cnt = 0
    while np.linalg.norm(Omega - Omega_old, ord=np.inf) > 0.001:
        print('--------', cnt+1, '--------')
        for i in range(n):
            if(i < l):
                Sigma[i, 0] = 1 / (1 + np.exp(-1 * np.dot(X[i, :], Omega)))
                E[i, i] = Sigma[i, 0] * (1 - Sigma[i, 0])
            else:
                Sigma[i, 0] = 1 / \
                    (1 + np.exp(-1 *
                                Lambda_vector[i-l, 0] * np.dot(X[i, :], Omega)))
                E[i, i] *= Mu * Lambda_vector[i-l, 0] * \
                    Lambda_vector[i-l, 0] * Sigma[i, 0] * (1 - Sigma[i, 0])
                Lambda[i, i] = Mu * Lambda_vector[i-l, 0]
                P[i-l, i-l] = np.dot(X[i, :], Omega)
                k_lambda[i-l, 0] = Beta * \
                    (1 - (1 / (1 + np.exp(-(Beta * Lambda_vector[i-l, 0])))))
                Eu[i-l, i-l] = Sigma[i, 0] * (1 - Sigma[i, 0])
                O[i-l, i-l] = Beta * Beta * (1 / (1 + np.exp(-(Beta * Lambda_vector[i-l, 0])))) * (
                    1 - (1 / (1 + np.exp(-(Beta * Lambda_vector[i-l, 0])))))
        if(np.linalg.norm(g_omega[:, 0], ord=2) / d) < 0.001:
            g_omega = np.dot(np.dot(X.T, Lambda), (y - Sigma)) - \
                np.dot((A + B), Omega)
            H_omega = -1 * (np.dot(np.dot(X.T, E), X) + A + B)
            Sig_omega = -1 * np.linalg.inv(H_omega)
            Omega_old = Omega.copy()
            Omega = Omega - np.dot(np.linalg.inv(H_omega), g_omega)
            print('gw:', np.mean(g_omega[:, 0]), ' gw_judge:', (np.linalg.norm(
                g_omega[:, 0], ord=2) / d), 'w_max', np.max(Omega, axis=0), 'w_min', np.min(Omega, axis=0))
        for i in range(d):
            if(Omega[i, 0] != 0) and (abs(Omega[i, 0]) < 0.001):
                Omega[i, 0] = 0
        if(np.linalg.norm(g_lambda[:, 0], ord=2) / u) < 0.001:
            g_lambda = Mu * np.dot(P, (y[l:] - Sigma[l:])) - \
                np.dot(C, Lambda_vector) + k_lambda
            H_lambda = -1 * ((Mu * np.dot(np.dot(P.T, Eu), P)) + C + O)
            Sig_lambda = -1 * np.linalg.inv(H_lambda)
            Lambda_vector_old = Lambda_vector.copy()
            Lambda_vector = Lambda_vector - \
                np.dot(np.linalg.inv(H_lambda), g_lambda)
            print('gl:', np.mean(g_lambda[:, 0]), ' gl_judge:', (np.linalg.norm(
                g_lambda[:, 0], ord=2) / u), 'l_max', np.max(Lambda_vector, axis=0), 'l_min', np.min(Lambda_vector, axis=0))
        for i in range(u):
            if(Lambda_vector[i, 0] != 0) and (abs(Lambda_vector[i, 0]) < 0.001):
                Lambda_vector[i, 0] = 0
        G = np.dot(np.dot(np.dot(np.linalg.inv(A), B), np.linalg.inv(
            np.matlib.identity(d) + np.dot(np.linalg.inv(A), B))), np.linalg.inv(A))
        for i in range(d):
            A[i, i] = 1 / (Omega[i, 0] * Omega[i, 0] +
                           G[i, i] + Sig_omega[i, i])
        for i in range(u):
            C[i, i] = 1 / (Lambda_vector[i, 0] *
                           Lambda_vector[i, 0] + Sig_lambda[i, i])
        print('max_lambda_new-old',
              np.linalg.norm(Lambda_vector - Lambda_vector_old, ord=np.inf))
        print('max_omega_new-old', np.linalg.norm(Omega - Omega_old, ord=np.inf))
        cnt += 1
        if cnt == 50:
            break
    # --- Test ---
    predict_y = np.zeros(testSize)
    predict_vector_y = np.dot(test_X, Omega).flatten()
    predict_vector_y *= CONFIG[name]['yScaler']
    threshold = CONFIG[name]['threshold']
    for i in range(testSize):
        if predict_vector_y[0, i] < threshold:
            predict_y[i] = -1
        else:
            predict_y[i] = 1
    print('predict_y:', predict_vector_y[0, :10])
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for idx in range(len(test_y)):
        if test_y[idx] == 1 and predict_y[idx] == 1:
            tp += 1
        elif test_y[idx] == 1 and predict_y[idx] == -1:
            fn += 1
        elif test_y[idx] == -1 and predict_y[idx] == 1:
            fp += 1
        elif test_y[idx] == -1 and predict_y[idx] == -1:
            tn += 1
    p = tp / (fp + tp)
    pf = fp / (fp + tn)
    pd = tp / (tp + fn)
    F_measure = 2 * pd * p / (pd + p)
    """ print('precision:', 100 * p, '%')
    print('recall:', 100 * recall_score(test_y, predict_y), '%')
    print('pf:', 100 * pf, '%')
    print('F-measure:', 100 * F_measure, '%')
    print('accuracy:', 100 * accuracy_score(test_y, predict_y), '%')
    print('AUC:', 100 * roc_auc_score(test_y, predict_y), '%') """
    print('precision:', p)
    print('recall:', recall_score(test_y, predict_y))
    print('pf:', pf)
    print('F-measure:', F_measure)
    print('accuracy:', accuracy_score(test_y, predict_y))
    print('AUC:', roc_auc_score(test_y, predict_y))


def ck_benchmark():
    # run learning methods in ck dataset
    for name in ck_name_field:
        # ['ant1', 'ivy2', 'jedit4', 'lucene2', 'synapse1', 'velocity1', 'xalan2']
        """ if name not in ['velocity1']:
            continue """
        train_name = f'{name}train'
        test_name = f'{name}test'
        trian_path = f'./data/CK/CKTrain/{train_name}.mat'
        test_path = f'./data/CK/CKTest/{test_name}.mat'
        # read train dataset
        data = scio.loadmat(trian_path)
        train_divided = []
        for key in data[f'{train_name}']:
            for k in key:
                train_divided.append(k)
        # read test dataset
        data = scio.loadmat(test_path)
        test_divided = []
        for key in data[f'{test_name}']:
            for k in key:
                test_divided.append(k)
        # 10% 20% 30%
        for i in range(0, 3):
            print('\n========= CK -', name, str(10+i*10) + '% ==========')
            # scramble data
            np.random.shuffle(train_divided[i])
            # get 10%/20%/30% train data
            X_train = train_divided[i][:, :20]
            y_train = train_divided[i][:, 20]
            # get 90%/80%/70% test data
            X_test = test_divided[i][:, :20]
            y_test = test_divided[i][:, 20]
            # preprocessing of abnormal label value
            if y_train.mean() > 1:
                for idx, (key) in enumerate(y_train):
                    if key == 2:
                        y_train[idx] = -1.0
            if y_test.mean() > 1:
                for idx, (key) in enumerate(y_test):
                    if key == 2:
                        y_test[idx] = -1.0
            # dataset expanding by SMOTE
            smo = SMOTE(random_state=0, k_neighbors=3)
            X_train, y_train = smo.fit_sample(X_train, y_train)
            # JSFS
            JSFS(X_train, y_train, X_test, y_test, name)


def nasa_benchmark():
    # run learning methods in NASA dataset
    for name in nasa_name_field:
        limit = {'cm1': 1, 'kc3': 30, 'mc2': 40, 'mw1': 0.4, 'pc1': 1, 'pc3': 1, 'pc4': 1, 'pc5': 1}
        """ if name not in ['pc5']:
            continue """
        train_name = f'{name}train'
        test_name = f'{name}test'
        trian_path = f'./data/NASA/NASATrain/{train_name}.mat'
        test_path = f'./data/NASA/NASATest/{test_name}.mat'
        # read train dataset
        data = scio.loadmat(trian_path)
        train_divided = []
        for key in data[f'{train_name}']:
            for k in key:
                train_divided.append(k)
        # read test dataset
        data = scio.loadmat(test_path)
        test_divided = []
        for key in data[f'{test_name}']:
            for k in key:
                test_divided.append(k)
        # 10% 20% 30%
        for i in range(0, 3):
            print('\n========= NASA -', name, str(10+i*10) + '% ==========')
            # scramble data
            np.random.shuffle(train_divided[i])
            # get 10%/20%/30% train data
            X_train = train_divided[i][:, :20]
            y_train = train_divided[i][:, 20]
            # get 90%/80%/70% test data
            X_test = test_divided[i][:, :20]
            y_test = test_divided[i][:, 20]
            # print(y_test)
            # preprocessing of abnormal label value
            if y_train.mean() > 1:
                for idx, (key) in enumerate(y_train):
                    if key == 2:
                        y_train[idx] = -1.0
            if y_test.mean() > 1:
                for idx, (key) in enumerate(y_test):
                    if key == 2:
                        y_test[idx] = -1.0
            # divide positive samples and negative samples
            for idx, (key) in enumerate(y_train):
                    if key < limit[name]:
                        y_train[idx] = -1.0
                    else:
                        y_train[idx] = 1.0
            for idx, (key) in enumerate(y_test):
                    if key < limit[name]:
                        y_test[idx] = -1.0
                    else:
                        y_test[idx] = 1.0
            # dataset expanding by SMOTE
            smo_1 = SMOTE(random_state=0,k_neighbors=3)
            smo_2 = SMOTE(random_state=0,k_neighbors=1)
            try:
                # X_train, y_train =  smo_1.fit_sample(X_train, y_train.astype('int'))
                X_train, y_train =  smo_1.fit_sample(X_train, y_train)
            except ValueError:
                X_train, y_train =  smo_2.fit_sample(X_train, y_train)
            # JSFS
            JSFS(X_train, y_train, X_test, y_test, name)


if __name__ == '__main__':
    ck_benchmark()
    nasa_benchmark()
