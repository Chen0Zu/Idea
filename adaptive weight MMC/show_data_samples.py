#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 23:21:31 2019

@author: chenzu
"""

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

file_path = './data/ORL database/ORL_32x32.mat'
data = scio.loadmat(file_path)
fea = data.get('fea')
himg = fea[0,:].reshape(32,32).T
plt.figure()
plt.axis('off')
for i in range(1,10):
    himg = np.c_[himg,fea[i,:].reshape(32,32).T]
plt.imshow(himg,'Greys_r')

himg = fea[10,:].reshape(32,32).T
plt.figure()
plt.axis('off')
for i in range(11,20):
    himg = np.c_[himg,fea[i,:].reshape(32,32).T]
plt.imshow(himg,'Greys_r')

file_path = './data/Yale database/Yale_32x32.mat'
data = scio.loadmat(file_path)
fea = data.get('fea')
himg = fea[0,:].reshape(32,32).T
plt.figure()
plt.axis('off')
for i in range(1,10):
    himg = np.c_[himg,fea[i,:].reshape(32,32).T]
plt.imshow(himg,'Greys_r')


himg = fea[10,:].reshape(32,32).T
plt.figure()
plt.axis('off')
for i in range(11,20):
    himg = np.c_[himg,fea[i,:].reshape(32,32).T]
plt.imshow(himg,'Greys_r')

file_path = './data/Extended Yale Face Database B/YaleB_32x32.mat'
data = scio.loadmat(file_path)
fea = data.get('fea')
himg = fea[0,:].reshape(32,32).T
plt.figure()
plt.axis('off')
for i in range(1,10):
    himg = np.c_[himg,fea[i,:].reshape(32,32).T]
plt.imshow(himg,'Greys_r')

himg = fea[10,:].reshape(32,32).T
plt.figure()
plt.axis('off')
for i in range(11,20):
    himg = np.c_[himg,fea[i,:].reshape(32,32).T]
plt.imshow(himg,'Greys_r')

himg = fea[64,:].reshape(32,32).T
plt.figure()
plt.axis('off')
for i in range(65,74):
    himg = np.c_[himg,fea[i,:].reshape(32,32).T]
plt.imshow(himg,'Greys_r')

himg = fea[74,:].reshape(32,32).T
plt.figure()
plt.axis('off')
for i in range(75,84):
    himg = np.c_[himg,fea[i,:].reshape(32,32).T]
plt.imshow(himg,'Greys_r')