#!/usr/bin/env python3

import numpy as np
import pickle
import BOGP

with open('HMOFDATA.pkl', 'rb') as f:
    HMOFDATA = pickle.load(f)

X_Physical = HMOFDATA['X_Physical']
X_Chemical = HMOFDATA['X_Chemical']
X_Topological = HMOFDATA['X_Topological']
Y = HMOFDATA['YAPI_Fixed']

n, m = Y.shape

""" set up feature matrix and test scores """
X = np.hstack((X_Physical, X_Chemical, X_Topological))
y = Y[:, 2]

""" ignore any MOFs with nan values - or suspicious 2 with very high y """
I = [i for i in range(n) if np.isnan(np.sum(X[i] + y[i])) == False and Y[i, 2] < 10]
y_true_values = y[I].reshape(-1, 1)

""" status vector """
STATUS = np.zeros((n, 1))


def return_y_value(i):
    return y_true_values[i]


Y = np.zeros((n, 1)) * np.nan
X = X[I]

n, d = X.shape

""" true top 100 to compare sample with """
top = np.argsort(y_true_valuesy)[-100:]

""" normalize features """
n, d = X.shape
for j in range(d):
    X[:, j] = X[:, j] - np.mean(X[:, j])
    sig = np.mean(X[:, j] ** 2) ** 0.5
    if sig > 1e-5:
        X[:, j] = X[:, j] / sig

""" sample  100 at random to start """
p = np.random.permutation(n)
nrand = 100
for i in range(nrand):
    STATUS[p[i]] = 2
    Y[p[i]] = return_y_value(p[i])

P = BOGP.prospector(X)

ntested = nrand

""" lets go! """
while ntested < 2000:
    P.fit(Y, STATUS)

    """ sample next point """
    ipick, kpick = P.pick_next(STATUS)

    """ show that we are testing ipick """
    STATUS[ipick, kpick] = 1

    """ now lets get the score and mark it as tested """

    Y[ipick, kpick] = return_y_value(ipick)
    STATUS[ipick, kpick] = 2

    """ count sample and print out current score """
    ntested = ntested + 1
    print(ntested)
    print(sum(1 for i in range(n) if i in top and STATUS[i, 0] == 2))
