# encoding: utf-8
"""
@author: zhou zelong
@contact: zzl850783164@163.com
@time: 2020/12/3 15:16
@file: withinProjectTest.py
@desc:
"""

from ImprovedSDA import ImprovedSDA
from DataProcess import load_data, train_data_process, test_data_process
from sklearn.metrics import precision_score, recall_score, f1_score


def run(data_train, data_test):
    X1, X2 = train_data_process(data_train)
    print(X1.shape, X2.shape)

    Y, label = test_data_process(data_test)
    isda = ImprovedSDA(X1, X2, Y, minSizeOfSubclass=10)
    predictions = isda.within_predict()
    print(precision_score(label, predictions))
    print(recall_score(label, predictions))
    print(f1_score(label, predictions))


if __name__ == '__main__':
    data_name_train = 'pc5train'
    filepath_train = './data/NASA/NASATrain/' + data_name_train + '.mat'
    data1_train, data2_train, data3_train = load_data(filepath_train, data_name_train)
    data_name_test = 'pc5test'
    filepath_test = './data/NASA/NASATest/' + data_name_test + '.mat'
    data1_test, data2_test, data3_test = load_data(filepath_test, data_name_test)
    run(data3_train, data3_test)