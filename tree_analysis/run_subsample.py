#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:42:09 2020

run a subsample screen to use in the tree analysos

@author: Hook
"""

import pickle
import BOGP
import numpy as np
import matplotlib.pyplot as plt

with open('split_data.pkl', 'rb') as f:
    SPLITDATA = pickle.load(f) 

X=SPLITDATA['TRAIN_X']
y=SPLITDATA['TRAIN_y_true']

n=len(X)

STATUS=np.zeros((n,1))
Y=np.zeros((n,1))*np.nan

top100=np.argsort(y)[-100:]

r=100
B=1000
f=10

p=np.random.permutation(n)
STATUS[p[:r]]=2
Y[p[:r]]=y[p[:r]].reshape(-1,1)
B=B-r

P=BOGP.prospector(X)
P.fit(Y,STATUS)
mu,var=P.predict()
plt.plot(mu,y,'.')
plt.show()

while B>0:
    P.fit(Y,STATUS)
    i=P.pick_next(STATUS)
    Y[i]=y[i]
    STATUS[i]=2
    B-=1
    print('B = '+str(B))
    print('R = '+str(sum(1 for i in top100 if STATUS[i]==2)))
    
tested=[i for i in range(n) if STATUS[i]==2]

SPLITDATA['ame_tested']=tested

with open('split_data_ame_tested.pkl', 'wb') as f:
    pickle.dump(SPLITDATA, f)

    