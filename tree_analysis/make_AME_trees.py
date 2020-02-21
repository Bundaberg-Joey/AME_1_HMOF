"""

makes trees either from AME output or random subsample of data

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

""" parameters """
tree_hyperparameters=tree_hyperparameters={'max_depth':5,'max_leaf_nodes':16,'class_weight':{0:1,1:1},'splitter':'random'}
Ngammas=20
GAMMA=np.linspace(1,10**2,Ngammas)
Nreps=10

""" find indices of vertices on lower convex hull - this is used to compute Pareto fronts """

def Lower_Convex_Hull(X):
    X=X/np.max(X,0)
    X=np.exp(8*X)
    ix=list(np.argsort(X[:,0]))
    X=X[ix,:]
    i=np.argmax(X[:,1]-X[:,0])
    Hull=[int(ix[i])]
    done=False
    G=np.zeros(len(X))
    while done==False:
        G=np.zeros(len(X))
        G[i+1:]=np.divide(X[i+1:,1]-X[i,1],X[i+1:,0]-X[i,0])
        G[np.isnan(G)]=np.inf
        if np.min(G[i+1:])<0:
            i=int(np.argmin(G[i+1:]))+i+1
            Hull.append(int(ix[i]))
        else:
            done=True
        if i==len(X)-1:
            done=True
    return Hull

""" takes full training data set , trains trees, finds pareto front, tests on test data """

def brute_tree(XTRAIN,istopTRAIN,XTEST,istopTEST):
    
    """ check that there is at least one +ve example in training set """
    
    ntrain=XTRAIN.shape[0]
    ntest=XTEST.shape[0]
    
    if np.sum(istopTRAIN)==0:
        return 0,[]

    cost0=np.zeros(Ngammas*Nreps)
    cost1=np.zeros(Ngammas*Nreps)
    cost0test=np.zeros(Ngammas*Nreps)
    cost1test=np.zeros(Ngammas*Nreps)
    
    precisionTRAIN=np.zeros(Ngammas*Nreps)
    precisionTEST=np.zeros(Ngammas*Nreps)
    recallTEST=np.zeros(Ngammas*Nreps)
    rate=np.zeros(Ngammas*Nreps)
    
    for iii in range(Ngammas):
               
        gamma=GAMMA[iii]
        
        for jjj in range(Nreps):
    
            """ train a tree using training data with random splitting """
            
            tree_hyperparameters['class_weight']={0:1,1:gamma}
            clf=tree.DecisionTreeClassifier(**tree_hyperparameters)
            clf.fit(XTRAIN,istopTRAIN)
            
            """" record costs and precision on validation data """
            
            pTRAIN=clf.predict(XTRAIN)
            precisionTRAIN[iii*Nreps+jjj]=np.divide(sum(1 for i in range(ntrain) if pTRAIN[i] == 1 and istopTRAIN[i]==1),sum(pTRAIN))
            cost0[iii*Nreps+jjj]=sum(1 for i in range(ntrain) if pTRAIN[i] == 1 and istopTRAIN[i]==0)
            cost1[iii*Nreps+jjj]=sum(1 for i in range(ntrain) if pTRAIN[i] == 0 and istopTRAIN[i]==1)
            
            """ record precision on test data """
            
            pTEST=clf.predict(XTEST)
            precisionTEST[iii*Nreps+jjj]=np.divide(sum(1 for i in range(ntest) if pTEST[i] == 1 and istopTEST[i]==1),sum(pTEST))
            recallTEST[iii*Nreps+jjj]=sum(1 for i in range(ntest) if pTEST[i] == 1 and istopTEST[i]==1)/sum(istopTEST)
            cost0test[iii*Nreps+jjj]=sum(1 for i in range(ntest) if pTEST[i] == 1 and istopTEST[i]==0)
            cost1test[iii*Nreps+jjj]=sum(1 for i in range(ntest) if pTEST[i] == 0 and istopTEST[i]==1)
            
            """ record positive rate on full data """
            
            rate[iii*Nreps+jjj]=(sum(pTRAIN)+sum(pTEST))/(ntrain+ntest)
    
    """ Compute Pareto front for validation data """
    
    Pareto = Lower_Convex_Hull(np.concatenate((cost0.reshape(-1,1),cost1.reshape(-1,1)),1))
    
    """ make some nice plots for whoever is watching """
    
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(cost0,cost1,'.')
    plt.plot(cost0[Pareto],cost1[Pareto],'d')
    plt.xlabel('errors on class zero training data')
    plt.ylabel('errors on class one training data')

    plt.subplot(122)
    plt.plot(cost0test,cost1test,'.')
    plt.plot(cost0test[Pareto],cost1test[Pareto],'d')
    plt.xlabel('errors on class zero test data')
    plt.ylabel('errors on class one test data')
    plt.show()
    
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.semilogy(precisionTRAIN,rate,'.')
    plt.semilogy(precisionTRAIN[Pareto],rate[Pareto],'d')
    plt.xlabel('precision on training data')
    plt.ylabel('positive rate')

    plt.subplot(132)    
    plt.semilogy(precisionTEST,rate,'.')
    plt.semilogy(precisionTEST[Pareto],rate[Pareto],'d')
    plt.xlabel('precision on test data')
    plt.ylabel('positive rate')

    plt.subplot(133)         
    plt.plot(precisionTEST,recallTEST,'.')
    plt.plot(precisionTEST[Pareto],recallTEST[Pareto],'d')
    plt.xlabel('precision on test data')
    plt.ylabel('recall on test data')
    plt.show()  
    
    return {'cost0':cost0,'cost1':cost1,'cost0test':cost0test,'cost1test':cost1test,'precisionTRAIN':precisionTRAIN,'precisionTEST':precisionTEST,'recallTEST':recallTEST,'rate':rate,'Pareto':Pareto}



""" takes probabilistic training data set, trains trees, finds pareto front, tests on test data and returns tree score """

def prob_tree(XTRAIN,probistopTRAIN,tested,istoptested,XTEST,istopTEST):
    
    ntrain=XTRAIN.shape[0]
    ntest=XTEST.shape[0]
    
    """ reformat training data as weighted deterministic """
    
    XTRAIN=np.concatenate((XTRAIN,XTRAIN),0)
    istopTRAIN=np.concatenate((np.ones(ntrain),np.zeros(ntrain)),0)
    weights=np.concatenate((probistopTRAIN,1-probistopTRAIN),0)
    for t in range(len(tested)):
        if istoptested[t]==1:
            weights[tested[t]]=1
            weights[tested[t]+ntrain]=0
        else:
            weights[tested[t]]=0
            weights[tested[t]+ntrain]=1

    cost0=np.zeros(Ngammas*Nreps)
    cost1=np.zeros(Ngammas*Nreps)
    cost0test=np.zeros(Ngammas*Nreps)
    cost1test=np.zeros(Ngammas*Nreps)
    
    precisionTRAIN=np.zeros(Ngammas*Nreps)
    precisionTEST=np.zeros(Ngammas*Nreps)
    recallTEST=np.zeros(Ngammas*Nreps)
    rate=np.zeros(Ngammas*Nreps)
    
    MODELS=[]
    
    for iii in range(Ngammas):
               
        gamma=GAMMA[iii]
        
        for jjj in range(Nreps):
            
            """ train a tree using training data with random splitting """
            
            tree_hyperparameters['class_weight']={0:1,1:gamma}
            clf=tree.DecisionTreeClassifier(**tree_hyperparameters)
            clf.fit(XTRAIN,istopTRAIN,weights)
            
            """" record costs and precision on training data """
            
            pTRAIN=clf.predict(XTRAIN)
            precisionTRAIN[iii*Nreps+jjj]=np.divide(sum(weights[i] for i in range(2*ntrain) if pTRAIN[i] == 1 and istopTRAIN[i]==1),sum(weights[i] for i in range(2*ntrain) if pTRAIN[i] == 1))
            cost0[iii*Nreps+jjj]=sum(weights[i] for i in range(2*ntrain) if pTRAIN[i] == 1 and istopTRAIN[i]==0)
            cost1[iii*Nreps+jjj]=sum(weights[i] for i in range(2*ntrain) if pTRAIN[i] == 0 and istopTRAIN[i]==1)
            MODELS.append(clf)
            
            """ record precision on test data """
            
            pTEST=clf.predict(XTEST)
            precisionTEST[iii*Nreps+jjj]=sum(1 for i in range(ntest) if pTEST[i] == 1 and istopTEST[i]==1)/sum(pTEST)
            recallTEST[iii*Nreps+jjj]=sum(1 for i in range(ntest) if pTEST[i] == 1 and istopTEST[i]==1)/sum(istopTEST)
            cost0test[iii*Nreps+jjj]=sum(1 for i in range(ntest) if pTEST[i] == 1 and istopTEST[i]==0)
            cost1test[iii*Nreps+jjj]=sum(1 for i in range(ntest) if pTEST[i] == 0 and istopTEST[i]==1)
            
            """ record positive rate on full data """
            
            rate[iii*Nreps+jjj]=(sum(pTRAIN)/2+sum(pTEST))/(ntrain+ntest)

    
    """ Compute Pareto front for validation data """
    
    Pareto = Lower_Convex_Hull(np.concatenate((cost0.reshape(-1,1),cost1.reshape(-1,1)),1))
    
    """ make some nice plots for whoever is watching """

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(cost0,cost1,'.')
    plt.plot(cost0[Pareto],cost1[Pareto],'d')
    plt.xlabel('errors on class zero training data')
    plt.ylabel('errors on class one training data')

    plt.subplot(122)
    plt.plot(cost0test,cost1test,'.')
    plt.plot(cost0test[Pareto],cost1test[Pareto],'d')
    plt.xlabel('errors on class zero test data')
    plt.ylabel('errors on class one test data')
    plt.show()
    
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.semilogy(precisionTRAIN,rate,'.')
    plt.semilogy(precisionTRAIN[Pareto],rate[Pareto],'d')
    plt.xlabel('precision on training data')
    plt.ylabel('positive rate')

    plt.subplot(132)    
    plt.semilogy(precisionTEST,rate,'.')
    plt.semilogy(precisionTEST[Pareto],rate[Pareto],'d')
    plt.xlabel('precision on test data')
    plt.ylabel('positive rate')

    plt.subplot(133)         
    plt.plot(precisionTEST,recallTEST,'.')
    plt.plot(precisionTEST[Pareto],recallTEST[Pareto],'d')
    plt.xlabel('precision on test data')
    plt.ylabel('recall on test data')
    plt.show()   
    
        
    return {'cost0':cost0,'cost1':cost1,'cost0test':cost0test,'cost1test':cost1test,'precisionTRAIN':precisionTRAIN,'precisionTEST':precisionTEST,'recallTEST':recallTEST,'rate':rate,'Pareto':Pareto,'MODELS':MODELS}