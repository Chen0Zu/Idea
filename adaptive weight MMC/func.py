#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:46:30 2019

@author: chenzu
"""

import numpy as np

def pca(X,d):
    U, sigma, VT = np.linalg.svd(X)
    return sigma[0:d], VT.T[:,0:d]

def lda(X,gnd,d):
    St = np.cov(X.T, bias = True)
    N,D = X.shape
    Sw = np.zeros([D,D])
    for i in np.unique(gnd):
        Sw = Sw + \
        sum(gnd.flatten()==i)/N*np.cov(X[gnd.reshape(-1) == i,:].T, bias = True)
    Sb = St - Sw
    
    U, sigma, VT = np.linalg.svd(np.linalg.inv(Sw)@Sb)
    return sigma[0:d], VT.T[:,0:d]

def mmc(X,gnd,d):
    St = np.cov(X.T, bias = True)
    N,D = X.shape
    Sw = np.zeros([D,D])
    for i in np.unique(gnd):
        Sw = Sw + \
        sum(gnd.flatten()==i)/N*np.cov(X[gnd.reshape(-1) == i,:].T, bias = True)
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
        Sw = Sw + \
        sum(gnd.flatten()==i)/N*np.cov(X[gnd.reshape(-1) == i,:].T, bias = True)
    Sb = St - Sw
    
    obj = np.zeros([Iter, 2])
    for i in range(Iter):
        # update projection vector
        U, sigma, VT = np.linalg.svd(alpha*Sb-alpha**2*Sw)
        eig_value = sigma[0:d]
        eig_vec = VT.T[:,0:d]
        
        # update alpha
        alpha = np.trace(eig_vec.T@Sb@eig_vec)/(2*np.trace(eig_vec.T@Sw@eig_vec))
        obj[i,0] = np.trace(alpha*(eig_vec.T@(Sb-alpha*Sw)@eig_vec))
        obj[i,1] = alpha
        
    return eig_value, eig_vec
