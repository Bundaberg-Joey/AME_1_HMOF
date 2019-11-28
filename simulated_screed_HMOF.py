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
num_initial_samples = 100
max_iterations = 2000

top = np.argsort(y_true)[-100:]  # true top 100 to compare sample with
STATUS = np.zeros((n, 1))  # column vector denoting status of material (0=untested, 1=testing, 2=tested)
y_experimental = np.full((n, 1), np.nan)  # column vector of determined material performances

initial_samples = np.random.randint(0, n, num_initial_samples)  # choose n values where each value is a material index
for sample in initial_samples:
    STATUS[sample] = 2
    y_experimental[sample] = determine_material_value(sample, y_true)  # determine material value and update

ntested = num_initial_samples

P = BOGP.prospector(X)

while ntested < max_iterations:  # lets go!
    P.fit(y_experimental, STATUS)
    ipick = P.pick_next(STATUS)  # sample next point
    STATUS[ipick, 0] = 1  # show that we are testing ipick
    y_experimental[ipick, 0] = determine_material_value(ipick, y_true)  # now lets get the score and update status
    STATUS[ipick, 0] = 2
    ntested = ntested + 1  # count sample and print out current score
    print(ntested)
    print(sum(1 for i in range(n) if i in top and STATUS[i, 0] == 2))
