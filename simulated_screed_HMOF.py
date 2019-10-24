#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:25:20 2019

tests BOGP for Bayesian Optimization on HMOF database
this is the non-belina version with nice plots etc... 

@author: Hook
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import BOGP

with open('HMOFDATA.pkl', 'rb') as f:
    HMOFDATA = pickle.load(f)
X_Physical=HMOFDATA['X_Physical']
X_Chemical=HMOFDATA['X_Chemical']
X_Topological=HMOFDATA['X_Topological']
Y=HMOFDATA['YAPI_Fixed']

n,m=Y.shape

""" set up feature matrix and test scores """
X=np.hstack((X_Physical,X_Chemical,X_Topological))
y=Y[:,2]

""" ignore any MOFs with nan values - or suspicious 2 with very high y """
I=[i for i in range(n) if np.isnan(np.sum(X[i]+y[i]))==False and Y[i,2]<10]
y=y[I]
X=X[I]

""" true top 100 to compare sample with """
top=np.argsort(y)[-100:]

""" normalize features """
n,d=X.shape
for j in range(d):
    X[:,j]=X[:,j]-np.mean(X[:,j])
    sig=np.mean(X[:,j]**2)**0.5
    if sig>1e-5:
        X[:,j]=X[:,j]/sig

""" sample  100 at random to start """
p=np.random.permutation(n)        
nrand=100
tested=list(p[:nrand])
untested=list(p[nrand:])
P=BOGP.prospector()
its_per_up=100
ntested=nrand

""" lets go! """
while ntested<2000:

    """ update hyperparameters every 100 samples """
    if np.mod(ntested,its_per_up)==0:
        P.fit(X,untested,tested,y[tested])
        """ estiamates number of top 100 found so far """
        """ plots actual top 100 vs estimaed along with 90% range """
        ny=100
        YS=P.samples(nsamples=ny)
        R=np.zeros((ntested,ny))
        for j in range(ny):
            topsapmled=np.argpartition(YS[:,j],-100)[-100:]
            R[:,j]=np.cumsum([i in topsapmled for i in tested])
        """ nice plot to show progress and estimates score """
        plt.plot(np.cumsum([i in top for i in tested]),label='true top 100')
        plt.plot(np.mean(R,1),label='estimated top 100')
        plt.plot([ntested,ntested],[np.sort(R[-1])[4],np.sort(R[-1])[-5]])
        plt.legend()
        plt.xlabel('number of testes')
        plt.ylabel('number of top 100 found')
        plt.show()        
  
    """ sample next point """
    i=P.pick_next(untested,acquisition_function='Thompson')
    tested.append(i)
    untested.remove(i)
    """ update model """
    P.update_model(tested,y[tested])
    
    """ count sample and print out current score """
    ntested=ntested+1
    print(ntested)
    print(sum(1 for i in tested if i in top))

""" save list of tested MOFs """
np.save('tested_thompson',tested)

""" some nice plots """

""" estiamates number of top 100 found so far """
""" plots actual top 100 vs estimaed along with 90% range """
ny=100
YS=P.samples(nsamples=ny)
R=np.zeros((ntested,ny))
for j in range(ny):
    topsapmled=np.argpartition(YS[:,j],-100)[-100:]
    R[:,j]=np.cumsum([i in topsapmled for i in tested])
""" nice plot to show progress and estimates score """
plt.plot(np.cumsum([i in top for i in tested]),label='true top 100')
plt.plot(np.mean(R,1),label='estimated top 100')
plt.plot([ntested,ntested],[np.sort(R[-1])[4],np.sort(R[-1])[-5]])
plt.legend()
plt.xlabel('number of testes')
plt.ylabel('number of top 100 found')
plt.show()    

""" breakdown into top 10, 25 etc... """
top1000=np.argsort(y)[-1000:]
N=[10,25,50,100,250,500]
for j in range(6):
    plt.plot(np.cumsum([i in top1000[-N[j]:] for i in tested])/N[j],label='top '+str(N[j]))
plt.legend()
plt.xlabel('number of testes')
plt.ylabel('proportion found')
plt.savefig('fig2.eps')
plt.show()

""" histogram of top MOFs found vs top MOFs in full database """
y1000=y[top1000[0]]
y100=y[top1000[-100]]
H=np.histogram(y[y>y1000])
plt.hist(y,H[1],label='full database')
plt.hist(y[tested],H[1],label='AME sample')
plt.plot([y100,y100],[0,np.max(H[0])],'--',color='black',label='top 100')
plt.legend()
plt.xlabel('API')
plt.ylabel('frequency')
plt.savefig('fig3.eps')
plt.show()
