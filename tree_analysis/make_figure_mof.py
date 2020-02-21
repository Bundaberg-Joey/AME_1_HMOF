#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:26:12 2019

make figure comparing AME trees to brute trees on HCOF deliverable capacity test data

@author: Hook
"""

import numpy as np
import matplotlib.pyplot as plt
import BOGP
import make_AME_trees
from scipy.stats import norm
import pickle


with open('split_data_ame_tested.pkl', 'rb') as f:
    SPLITDATA = pickle.load(f) 

TEST_X=SPLITDATA['TEST_X']
TEST_y_true=SPLITDATA['TEST_y_true']
TRAIN_X=SPLITDATA['TRAIN_X']
TRAIN_y_true=SPLITDATA['TRAIN_y_true']
ame_tested=SPLITDATA['ame_tested']

tau=8.369 # chose to find top500 

ntrain=len(TRAIN_X)
STATUS=np.zeros((ntrain,1))
Y=np.zeros((ntrain,1))*np.nan
for i in ame_tested:
    STATUS[i]=2
    Y[i]=TRAIN_y_true[i]

""" calcaulte posteriori probabiltiy of being over tau """
P=BOGP.prospector(TRAIN_X)
P.fit(Y,STATUS)
mu,var=P.predict()
probistopTRAIN=1-norm.cdf(np.divide(tau-mu.reshape(ntrain),var.reshape(ntrain)**0.5))

istopTRAIN=(TRAIN_y_true>tau)*1
istopTESTED=(TRAIN_y_true[ame_tested]>tau)*1
istopTESTSET=(TEST_y_true>tau)*1

AMEtrees=make_AME_trees.prob_tree(TRAIN_X,probistopTRAIN,ame_tested,istopTESTED,TEST_X,istopTESTSET)

BRUTEtrees=[]
for k in [1,0.25,0.1,0.05,0.01]:
    nsample=int(ntrain*k)
    p=np.random.permutation(ntrain)
    sample=list(p[:nsample])
    BRUTEtrees.append(make_AME_trees.brute_tree(TRAIN_X[sample],istopTRAIN[sample],TEST_X,istopTESTSET))
    


cost0,cost1,cost0test,cost1test,precisionTRAIN,precisionTEST,recallTEST,rate,Pareto,MODELS=list(AMEtrees.values())
plt.plot(precisionTEST[Pareto],recallTEST[Pareto],'s',label='AME')
lab=['Brute Force','Random Subsample $25\%$','Random Subsample $10\%$','Random Subsample $5\%$','Random Subsample $1\%$']
i=0
cost0,cost1,cost0test,cost1test,precisionTRAIN,precisionTEST,recallTEST,rate,Pareto=list(BRUTEtrees[i].values())
plt.plot(precisionTEST[Pareto],recallTEST[Pareto],'d',label=lab[i])
plt.legend()
plt.xlabel('precision on test data')
plt.ylabel('recall on test data')
plt.savefig('figures/precision_vs_recall_HCOF_DC_0.png')
plt.show()
for t in range(4):
    cost0,cost1,cost0test,cost1test,precisionTRAIN,precisionTEST,recallTEST,rate,Pareto,MODELS=list(AMEtrees.values())
    plt.plot(precisionTEST[Pareto],recallTEST[Pareto],'s',label='AME')
    lab=['Brute Force','Random Subsample $25\%$','Random Subsample $10\%$','Random Subsample $5\%$','Random Subsample $1\%$']
    i=0
    cost0,cost1,cost0test,cost1test,precisionTRAIN,precisionTEST,recallTEST,rate,Pareto=list(BRUTEtrees[i].values())
    plt.plot(precisionTEST[Pareto],recallTEST[Pareto],'d',label=lab[i])
    for i in range(1,2+t):
        cost0,cost1,cost0test,cost1test,precisionTRAIN,precisionTEST,recallTEST,rate,Pareto=list(BRUTEtrees[i].values())
        plt.plot(precisionTEST[Pareto],recallTEST[Pareto],'o',label=lab[i])
    plt.legend()
    plt.xlabel('precision on test data')
    plt.ylabel('recall on test data')
    plt.savefig('figures/precision_vs_recall_HCOF_DC_'+str(t+1)+'.png')
    plt.show()


cost0,cost1,cost0test,cost1test,precisionTRAIN,precisionTEST,recallTEST,rate,Pareto,MODELS=list(AMEtrees.values())
plt.semilogy(precisionTEST[Pareto],rate[Pareto],'d',label='AME')
lab=['100','25','10','5','1']
for i in range(5):
    cost0,cost1,cost0test,cost1test,precisionTRAIN,precisionTEST,recallTEST,rate,Pareto=list(BRUTEtrees[i].values())
    plt.semilogy(precisionTEST[Pareto],rate[Pareto],'d',label=lab[i])
plt.legend()
plt.xlabel('precision on test data')
plt.ylabel('positive rate')
plt.ylim([10**-3,10**-1])
plt.savefig('figures/precision_vs_rate_HCOF_DC.png')
plt.show()


cost0,cost1,cost0test,cost1test,precisionTRAIN,precisionTEST,recallTEST,rate,Pareto,MODELS=list(AMEtrees.values())
plt.plot(cost0test[Pareto],cost1test[Pareto],'d',label='AME')
lab=['100','25','10','5','1']
for i in range(5):
    cost0,cost1,cost0test,cost1test,precisionTRAIN,precisionTEST,recallTEST,rate,Pareto=list(BRUTEtrees[i].values())
    plt.plot(cost0test[Pareto],cost1test[Pareto],'d',label=lab[i])
plt.legend()
plt.xlabel('errors on class 0 test data')
plt.ylabel('errors on class 1 test data')
plt.savefig('figures/cost1_vs_cost2_HCOF_DC.png')
plt.show()


# picks out 8th from right tree model
model=AMEtrees['MODELS'][AMEtrees['Pareto'][7]]
""" print a graph of this tree somehow! """
