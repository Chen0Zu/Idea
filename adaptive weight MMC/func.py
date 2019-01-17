#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:46:30 2019

@author: chenzu
"""

import numpy as np

def pca(X,d):
    St = np.cov(X.T， bias = True)
    eig_value, eig_vec = np.linalg.eig(St)
    idx = np.argsort(-eig_value)
    
    return eig_value[idx[0:d+1]], eig_vec[:,idx[0:d+1]]

def lda(X,gnd,d):
    St = np.cov(X.T, bias = True)
    N,D = X.shape
    Sw = np.zeros([D,D])
    for i in np.unique(gnd):
        Sw = Sw + \
        sum(gnd.flatten()==i)/N*np.cov(X[gnd.reshape(-1) == i,:].T, bias = True)
    Sb = St - Sw
    
    eig_value, eig_vec = np.linalg.eig(np.linalg.inv(Sw)@Sb)
    idx = np.argsort(-eig_value)
    return eig_value[idx[0:d+1]], eig_vec[:,idx[0:d+1]]

def mmc(X,gnd,d):
    St = np.cov(X.T, bias = True)
    N,D = X.shape
    Sw = np.zeros([D,D])
    for i in np.unique(gnd):
        Sw = Sw + sum(gnd.flatten()==i)/N*np.cov(X[gnd.reshape(-1) == i,:].T, bias = True)
    Sb = St - Sw
    
    eig_value, eig_vec = np.linalg.eig(Sb-Sw)
    idx = np.argsort(-eig_value)
    return eig_value[idx[0:d+1]], eig_vec[:,idx[0:d+1]]

def weightmmc(X,gnd,d,alpha = 1):
    Iter = 10
    
    St = np.cov(X.T,bias=True)
    N,D = X.shape
    Sw = np.zeros([D,D])
    for i in np.unique(gnd):
        Sw = Sw + sum(gnd==i)/N*np.cov(X[gnd.reshape(-1) == i,:].T, bias = True)
    Sb = St - Sw
    
    obj = np.zeros([Iter, 2])
    for i in range(Iter):
        # update projection vector
        eig_value, eig_vec = np.linalg.eig(alpha*Sb-alpha**2*Sw)
        idx = np.argsort(-eig_value)
        eig_value = eig_value[idx[0:d]]
        eig_vec = eig_vec[:,idx[0:d]]
        
        # update alpha
        alpha = np.trace(eig_vec.T@Sb@eig_vec)/(2*np.trace(eig_vec.T@Sw@eig_vec))
        obj[i,0] = np.trace(alpha*(eig_vec.T@(Sb-alpha*Sw)@eig_vec))
        obj[i,1] = alpha
        
    return obj
