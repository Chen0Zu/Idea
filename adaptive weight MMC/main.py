#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:21:33 2019

@author: chenzu
"""

import numpy as np
import matplotlib.pyplot as plt

from func import mmc
from func import weightmmc

mu1 = np.array([0,0,0,0])
mu2 = np.array([1,1,1,1])
cov = np.diag([1,1,1,1])
N = 300

X1 = np.random.multivariate_normal(mu1, cov, int(N/2))
X2 = np.random.multivariate_normal(mu2, cov, int(N/2))
X = np.r_[X1, X2]
gnd = np.r_[np.ones([int(N/2),1]), -np.ones([int(N/2),1])]

a,b = mmc(X,gnd,1)
obj = weightmmc(X,gnd,2,40)
obj2 = weightmmc(X,gnd,3,40)
