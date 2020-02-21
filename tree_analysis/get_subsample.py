#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:42:09 2020

split out a test set for tree analysis

@author: Hook
"""

import pickle
import numpy as np

with open('../../HMOFDATA.pkl', 'rb') as f:
    HMOFDATA = pickle.load(f) 

y_true=HMOFDATA['YAPI_Fixed'][:,2]
#X=np.hstack((HMOFDATA['X_Physical'],HMOFDATA['X_Chemical'],HMOFDATA['X_Topological']))
X=np.hstack((HMOFDATA['X_Physical'],HMOFDATA['X_Chemical']))

n,d=X.shape

ISOK=[i for i in range(n) if np.sum(np.isnan(X[i]))==0 and np.isnan(y_true[i])==0]

X=X[ISOK]
y_true=y_true[ISOK]

for j in range(d):
    X[:,j]=X[:,j]-np.mean(X[:,j])
    std=np.mean(X[:,j]**2)**0.5
    if std>1e-6:
        X[:,j]=X[:,j]/std

nok=X.shape[0]

ntest=int(nok*0.1)
ntrain=nok-ntest

p=np.random.permutation(nok)

TEST_I=[ISOK[i] for i in p[:ntest]]
TRAIN_I=[ISOK[i] for i in p[ntest:]]

TEST_X=X[p[:ntest]]
TEST_y_true=y_true[p[:ntest]]

TRAIN_X=X[p[ntest:]]
TRAIN_y_true=y_true[p[ntest:]]

with open('split_data.pkl', 'wb') as f:
    pickle.dump({'TEST_I':TEST_I,'TRAIN_I':TRAIN_I,'TEST_X':TEST_X,'TEST_y_true':TEST_y_true,'TRAIN_X':TRAIN_X,'TRAIN_y_true':TRAIN_y_true}, f)
