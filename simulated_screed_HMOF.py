#!/usr/bin/env python3

import numpy as np
import BOGP

########################################################################################################################


def data_loader(path_to_file):
    data_set = np.loadtxt(path_to_file)
    X, y = data_set[:, :-1], data_set[:, -1]
    y = y.reshape(-1, 1)
    n, d = X.shape
    return X, y, n, d


########################################################################################################################


def determine_material_value(material, true_results):
    return true_results[material]


########################################################################################################################

X, y_true, n, d = data_loader(r'C:\Users\crh53\OneDrive\Desktop\PHD_Experiments\E2_AMI_James\Data\Scaled_HMOF_Data')

top = np.argsort(y_true)[-100:]  # true top 100 to compare sample with
STATUS = np.zeros((n, 1))  # column vector denoting status of material (0=untested, 1=testing, 2=tested)
y_experimental = np.full((n, 1), np.nan)  # column vector of determined material performances

material_indices = np.random.permutation(n)  # sample  100 at random to start
nrand = 100
ntested = nrand
for i in range(nrand):
    STATUS[material_indices[i]] = 2
    y_experimental[material_indices[i]] = determine_material_value(material_indices[i], y_true)

P = BOGP.prospector(X)

while ntested < 2000:  # lets go!
    P.fit(y_experimental, STATUS)
    ipick, kpick = P.pick_next(STATUS)  # sample next point
    STATUS[ipick, kpick] = 1  # show that we are testing ipick
    y_experimental[ipick, kpick] = determine_material_value(ipick, y_true)  # now lets get the score and update status
    STATUS[ipick, kpick] = 2
    ntested = ntested + 1  # count sample and print out current score
    print(ntested)
    print(sum(1 for i in range(n) if i in top and STATUS[i, 0] == 2))
