# encoding: utf-8
"""
@author: zhou zelong
@contact: zzl850783164@163.com
@time: 2020/12/3 15:16
@file: ImprovedSDA.py
@desc:
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import scipy.io
import scipy.linalg


# done!
def get_skewed_F_measure(X1, X2, h1, h2, alpha):
    TP = 0  # the number of defective instances that are predicted as defective (true judgment)
    TN = 0  # the number of defect-free instances that are predicted as defect-free (true judgment)
    FP = 0  # the number of defect-free instances that are predicted as defective (misjudgement)
    FN = 0  # the number of defective instances that are predicted as defect-free (misjudgement)

    # predicted defective instances
    n1 = len(X1)
    for i in range(0, n1):
        # y = X1.pop(i)
        y = X1[i]
        X1 = np.delete(X1, i, axis=0)
        if get_label_of_y(X1, X2, h1, h2, y):
            TP = TP + 1
        else:
            FN = FN + 1
        # X1.insert(i, y)
        X1 = np.insert(X1, i, y, axis=0)

    # predicted defect-free instances
    n2 = len(X2)
    for i in range(0, n2):
        # y = X2.pop(i)
        y = X2[i]
        X2 = np.delete(X2, i, axis=0)
        if get_label_of_y(X1, X2, h1, h2, y):
            FP = FP + 1
        else:
            TN = TN + 1
        # X2.insert(i, y)
        X2 = np.insert(X2, i, y, axis=0)

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Pf = FP / (FP + TN)  # false positive rate
    TNR = 1 - Pf  # true negative rate
    skewedFMeasure = (1 + alpha) * Precision * Recall / (alpha * Precision + Recall)
    return skewedFMeasure


# done!
def get_H1_H2_for_I_SDA(X1, X2, minSizeOfSubclass):
    n1 = len(X1)
    n2 = len(X2)
    H1 = 0
    H2 = 0
    skewedFMeasure = -1
    subSize = minSizeOfSubclass
    while subSize*2 <= n1 and subSize*2 <= n2:
        h1 = round(n1 / subSize)
        h2 = round(n2 / subSize)
        sf = get_skewed_F_measure(X1, X2, h1, h2, 4)
        if skewedFMeasure < sf:
            skewedFMeasure = sf
            H1 = h1
            H2 = h2
        print(('h1 = %d' % h1, 'subSize1 = %d' % (n1/h1)), ('h2 = %d' % h2, 'subSize2 = %d' % (n2/h2)), 'sf = %f' % sf)
        subSize = subSize + 1
    return H1, H2


# done!
def sort_for_nnc(X):
    n, m = X.shape
    sortedX = np.zeros(shape=(n, m))
    euclideanDistance = np.zeros(shape=(n, n))
    maxDistance = -1
    s = 0
    b = 0
    for i in range(0, n):
        for j in range(i + 1, n):
            euclideanDistance[i][j] = np.linalg.norm(X[i] - X[j])
            euclideanDistance[j][i] = euclideanDistance[i][j]
            if maxDistance < euclideanDistance[i][j]:
                maxDistance = euclideanDistance[i][j]
                s = i
                b = j
    sortedX[0] = X[s]
    sortedX[n-1] = X[b]
    euclideanDistance[s][b] = float('inf')
    euclideanDistance[b][s] = float('inf')
    for g in range(0, int((n-1)/2)):
        minDistance = float('inf')
        m = 0
        for j in range(0, n):
            if euclideanDistance[s][j] < minDistance and j != s:
                minDistance = euclideanDistance[s][j]
                m = j
        sortedX[g+1] = X[m]
        euclideanDistance[s][m] = float('inf')
        euclideanDistance[b][m] = float('inf')

        if g+1 != n-g-2:
            minDistance = float('inf')
            k = 0
            for j in range(0, n):
                if euclideanDistance[b][j] < minDistance and j != b:
                    minDistance = euclideanDistance[b][j]
                    k = j
            sortedX[n-g-2] = X[k]
            euclideanDistance[s][k] = float('inf')
            euclideanDistance[b][k] = float('inf')
    return sortedX


# done!
def NNC(X1, X2, H1, H2):
    sortedX1 = sort_for_nnc(X1)
    sortedX2 = sort_for_nnc(X2)
    subX1 = np.array_split(sortedX1, H1)
    subX2 = np.array_split(sortedX2, H2)
    return subX1, subX2


# done!
# subX1 : list of np.array
def get_sumB(subX1, subX2, n1, n2):
    n = n1 + n2  # the number of all samples
    H1 = len(subX1)
    H2 = len(subX2)
    sum_B = 0
    for i in range(H1):
        p_1i = len(subX1[i]) / n
        u_1i = np.mean(subX1[i], axis=0)
        u_1i = np.array([u_1i.tolist()])
        for j in range(H2):
            p_2j = len(subX2[j]) / n
            u_2j = np.mean(subX2[j], axis=0)
            u_2j = np.array([u_2j.tolist()])
            gap = u_1i-u_2j
            sum_B = sum_B + (p_1i * p_2j * np.dot(gap.T, gap))  # different with the paper
    return sum_B


# done!
def get_sumX(X1, X2, n1, n2):
    u1 = np.mean(X1, axis=0)
    u2 = np.mean(X2, axis=0)
    u = (u1 + u2) / 2
    u = np.array([u.tolist()])
    sum_X = 0
    for i in range(n1):
        x = np.array([X1[i].tolist()])
        gap = x - u
        sum_X = sum_X + np.dot(gap, gap.T)  # different with the paper
    for i in range(n2):
        x = np.array([X2[i].tolist()])
        gap = x - u
        sum_X = sum_X + np.dot(gap.T, gap)  # different with the paper
    return sum_X


# done!
def get_V(sumB, sumX):
    sumX_inv = np.linalg.pinv(sumX)
    dot = np.dot(sumX_inv, sumB)
    w, v = np.linalg.eig(dot)
    v = v.T
    len_w = len(w)
    V = []
    for i in range(len_w):
        if w[i] != 0:
            V.append(v[1])
    return np.real(np.array(V))


# done!
def get_label_of_y(X1, X2, h1, h2, y):
    if len(y.shape) == 1:
        y = np.array([y.tolist()])
    n1 = len(X1)
    n2 = len(X2)
    # step 2
    subX1, subX2 = NNC(X1, X2, h1, h2)
    # step 3
    sum_B = get_sumB(subX1, subX2, n1, n2)
    sum_X = get_sumX(X1, X2, n1, n2)
    # step 4
    V = get_V(sum_B, sum_X)
    # step 5
    X = np.concatenate((X1, X2), axis=0)
    X_f = np.dot(V.T, X.T)  # different with the paper
    y_f = np.dot(V.T, y.T)  # different with the paper
    # step 6
    labels1 = np.ones(n1, dtype='i4')
    labels2 = np.zeros(n2, dtype='i4')
    labels = np.concatenate((labels1, labels2), axis=0)
    rf = RandomForestClassifier()
    rf.fit(X_f.T, labels)
    predictions = rf.predict(y_f.T)
    return predictions


# done!
def kernel(X1, X2, ker='primal', gamma=1):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


def TCA(Xs, Xt, kernel_type='primal', dim=16, lamb=1, gamma=1):
    """
    Transform Xs and Xt
    :param gamma: kernel bandwidth for rbf kernel
    :param lamb: lambda value in equation
    :param dim: dimension after transfer
    :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
    :param Xs: ns * n_feature, source feature
    :param Xt: nt * n_feature, target feature
    :return: Xs_new and Xt_new after TCA
    """
    X = np.hstack((Xs.T, Xt.T))
    X = X / np.linalg.norm(X, axis=0)
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
    M = e * e.T
    M = M / np.linalg.norm(M, 'fro')
    H = np.eye(n) - 1 / n * np.ones((n, n))
    K = kernel(X, None, ker=kernel_type, gamma=gamma)
    n_eye = m if kernel_type == 'primal' else n
    a, b = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
    w, V = scipy.linalg.eig(a, b)
    ind = np.argsort(w)
    A = V[:, ind[:dim]]
    Z = np.dot(A.T, K)
    Z = Z / np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
    return Xs_new, Xt_new


# done!
def within_project_ISDA(X1, X2, y, minSizeOfSubclass):
    n1 = len(X1)
    n2 = len(X2)
    print(('n1 = %d' % n1, 'n2 = %d' % n2))
    # step 1
    # H1, H2 = get_H1_H2_for_I_SDA(X1, X2, minSizeOfSubclass)
    H1 = round(n1 / minSizeOfSubclass)
    H2 = round(n2 / minSizeOfSubclass)
    print(('H1 = %d' % H1, 'subSize1 = %d' % (n1/H1)), ('H2 = %d' % H2, 'subSize2 = %d' % (n2/H2)))
    # step 2 - 6
    predictions = get_label_of_y(X1, X2, H1, H2, y)
    return predictions


# done!
def cross_project_SSTCA_ISDA(Xs1, Xs2, Xt, minSizeOfSubclass):
    n1 = len(Xs1)
    n2 = len(Xs2)
    Xs = np.vstack((Xs1, Xs2))
    # step 1
    Xs_new, Xt_new = TCA(Xs, Xt)
    Xs_new1 = Xs_new[: n1]
    Xs_new2 = Xs_new[n1: n1+n2]
    # step 2 - 3
    predictions = within_project_ISDA(Xs_new1, Xs_new2, Xt, minSizeOfSubclass)
    return predictions


class ImprovedSDA:
    def __init__(self, X1, X2, Xt, minSizeOfSubclass):
        """
        :param X1: n1 * n_feature, defective instances
        :param X2: n2 * n_feature, defect-free instances
        :param Xt: nt * n_feature, unlabeled instances
        :param minSizeOfSubclass:
        """
        self.X1 = X1
        self.X2 = X2
        self.Xt = Xt
        self.minSizeOfSubclass = minSizeOfSubclass

    def within_predict(self):
        return within_project_ISDA(self.X1, self.X2, self.Xt, self.minSizeOfSubclass)

    def cross_predict(self):
        return cross_project_SSTCA_ISDA(self.X1, self.X2, self.Xt, self.minSizeOfSubclass)
