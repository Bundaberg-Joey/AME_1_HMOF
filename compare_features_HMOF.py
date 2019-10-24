#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 20:48:19 2019

uses GPy

@author: Hook
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import GPy

with open('HMOFDATA.pkl', 'rb') as f:
    HMOFDATA = pickle.load(f)
    
X_Physical=HMOFDATA['X_Physical']
X_Chemical=HMOFDATA['X_Chemical']
X_Topological=HMOFDATA['X_Topological']

Y=HMOFDATA['YAPI_Fixed']


n,m=Y.shape

summed=np.sum(np.hstack((X_Physical,X_Chemical,X_Topological,Y)),1)
I=[i for i in range(n) if np.isnan(summed[i])==False and Y[i,2]<10]

Y=Y[I]
X_Physical=X_Physical[I]
X_Chemical=X_Chemical[I]
X_Topological=X_Topological[I]

X_Physical[:,-1]=np.log(X_Physical[:,-1])
X_Chemical=np.log(0.01+X_Chemical)
X_Topological=np.log(0.001+X_Topological)
    
for X in [X_Physical,X_Chemical,X_Topological]:
    n,d=X.shape
    for j in range(d):
        X[:,j]=X[:,j]-np.mean(X[:,j])
        sig=np.mean(X[:,j]**2)**0.5
        if sig>1e-5:
            X[:,j]=X[:,j]/sig
            
XX=[X_Physical,X_Chemical,X_Topological,np.hstack((X_Topological,X_Chemical)),np.hstack((X_Physical,X_Topological)),np.hstack((X_Physical,X_Chemical)),np.hstack((X_Physical,X_Chemical,X_Topological))]

ntrain=500
nfolds=10

RRMSE=np.zeros((nfolds,8,7))

n=len(I)

for jjj in range(nfolds):
    p=np.random.permutation(n)
    train=p[:ntrain]
    test=p[ntrain:]
    for iii in range(8):
        for k in range(7):
            X=XX[k]
            y=Y[:,iii].reshape(-1,1)
            n,d=X.shape
            mfy=GPy.mappings.Constant(input_dim=d,output_dim=1)
            ky = GPy.kern.RBF(d,ARD=True,lengthscale=np.ones(d))
            GP = GPy.models.GPRegression(X[train],y[train],kernel=ky,mean_function=mfy)
            GP.optimize('bfgs')
            z=GP.predict(X)[0]
            RRMSE[jjj,iii,k]=(np.mean((z[test]-y[test])**2)/np.cov(y[test].reshape(-1)))**0.5
            print(RRMSE[jjj])
            plt.plot(z,y,'.')
            plt.show()
            
            np.save('RRMSE',RRMSE)
    
