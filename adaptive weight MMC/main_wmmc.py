#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:21:33 2019

@author: chenzu
"""

import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn import neighbors
#from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

from func import weightmmc
# =============================================================================
# knn ORL数据集分类
# =============================================================================
file_path = './data/ORL database/ORL_32x32.mat'
data = scio.loadmat(file_path)
fea = data.get('fea')
gnd = data.get('gnd')
d = 100
repeat = 10
acc = np.zeros([repeat,1])
for i in range(repeat):
    train_X, test_X, train_gnd, test_gnd = \
    train_test_split(fea, gnd, test_size = 0.5, random_state=i, stratify=gnd)
#    scaler = StandardScaler().fit(train_X)
#    train_X = scaler.transform(train_X)
#    test_X = scaler.transform(test_X)
    mu = np.mean(train_X,0).reshape(-1,1).T
    train_X = train_X - mu
    test_X = test_X - mu
    eig_value, eig_vec = weightmmc(train_X, train_gnd, d)
    clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(train_X@eig_vec, train_gnd.flatten())
    label = clf.predict(test_X@eig_vec)
    acc[i] = np.mean(label == test_gnd.flatten())

acc_knn_orl = np.mean(acc)
var_orl = np.std(acc)

# =============================================================================
# knn yale数据及分类
# =============================================================================
file_path = './data/Yale database/Yale_32x32.mat'
data = scio.loadmat(file_path)
fea = data.get('fea')
gnd = data.get('gnd')
d = 200
repeat = 10
acc = np.zeros([repeat,1])
for i in range(repeat):
    train_X, test_X, train_gnd, test_gnd = \
    train_test_split(fea, gnd, test_size = 0.5, random_state=i, stratify=gnd)
#    scaler = StandardScaler().fit(train_X)
#    train_X = scaler.transform(train_X)
#    test_X = scaler.transform(test_X)
    mu = np.mean(train_X,0).reshape(-1,1).T
    train_X = train_X - mu
    test_X = test_X - mu
    eig_value, eig_vec = weightmmc(train_X, train_gnd, d)
    clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(train_X@eig_vec, train_gnd.flatten())
    label = clf.predict(test_X@eig_vec)
    acc[i] = np.mean(label == test_gnd.flatten())

acc_knn_yale = np.mean(acc)
var_yale = np.std(acc)

# =============================================================================
# knn extended yale数据及分类
# =============================================================================
file_path = './data/Extended Yale Face Database B/YaleB_32x32.mat'
data = scio.loadmat(file_path)
fea = data.get('fea')
gnd = data.get('gnd')
d = 100
repeat = 10
acc = np.zeros([repeat,1])
for i in range(repeat):
    train_X, test_X, train_gnd, test_gnd = \
    train_test_split(fea, gnd, test_size = 0.5, random_state=i, stratify=gnd)
#    scaler = StandardScaler().fit(train_X)
#    train_X = scaler.transform(train_X)
#    test_X = scaler.transform(test_X)
    mu = np.mean(train_X,0).reshape(-1,1).T
    train_X = train_X - mu
    test_X = test_X - mu
    eig_value, eig_vec = weightmmc(train_X, train_gnd, d)
    clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(train_X@eig_vec, train_gnd.flatten())
    label = clf.predict(test_X@eig_vec)
    acc[i] = np.mean(label == test_gnd.flatten())

acc_knn_yaleB = np.mean(acc)
var_yaleB = np.std(acc)