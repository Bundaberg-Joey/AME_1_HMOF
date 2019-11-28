#!/usr/bin/env python3

import numpy as np
import pickle
import BOGP


def return_y_value(i):
    return y_true_values[i]


data_set = np.loadtxt(r'C:\Users\crh53\OneDrive\Desktop\PHD_Experiments\E2_AMI_James\Data\Scaled_HMOF_Data')
X, y = data_set[:, :-1], data_set[:, -1]
y_true_values = y.reshape(-1, 1)
n, d = X.shape

STATUS = np.zeros((n, 1))  # status vector
Y = np.zeros((n, 1)) * np.nan


top = np.argsort(y_true_values)[-100:]  # true top 100 to compare sample with


p = np.random.permutation(n)  # sample  100 at random to start
nrand = 100
ntested = nrand
for i in range(nrand):
    STATUS[p[i]] = 2
    Y[p[i]] = return_y_value(p[i])

P = BOGP.prospector(X)

while ntested < 2000:  # lets go!
    P.fit(Y, STATUS)
    ipick, kpick = P.pick_next(STATUS)  # sample next point
    STATUS[ipick, kpick] = 1  # show that we are testing ipick
    Y[ipick, kpick] = return_y_value(ipick)  # now lets get the score and mark it as tested
    STATUS[ipick, kpick] = 2
    ntested = ntested + 1  # count sample and print out current score
    print(ntested)
    print(sum(1 for i in range(n) if i in top and STATUS[i, 0] == 2))
