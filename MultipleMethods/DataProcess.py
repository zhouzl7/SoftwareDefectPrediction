# encoding: utf-8
"""
@author: zhou zelong
@contact: zzl850783164@163.com
@time: 2020/12/3 15:16
@file: DataProcess.py
@desc: 
"""

from scipy.io import loadmat
import numpy as np


def load_data(filepath, data_name):
    data = loadmat(filepath)[data_name]
    return data[:, 0][0], data[:, 1][0], data[:, 2][0]


def train_data_process(data):
    n, m = data.shape
    return data[:, 0:m - 1], (data[:, m - 1] > 0).astype(int)


def test_data_process(data):
    n, m = data.shape
    return data[:, 0:m-1], (data[:, m-1] > 0).astype(int)
