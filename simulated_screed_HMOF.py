#!/usr/bin/env python3

import numpy as np
import BOGP

########################################################################################################################


def data_loader(path_to_file):
    """
    Loads the features and target variables for the AME to assess. Currently set up as numpy array loading but can
    easily retrofit depending on final decided file type

    :param path_to_file: str, location of the Feature_target file
    :return: features: np.array(), `m` by 'n' array which is the feature matrix of the data being modelled
    :return: labels: np.array(), `m` sized array containing the target values for the passed features
    """
    data_set = np.loadtxt(path_to_file)
    features, labels = data_set[:, :-1], data_set[:, -1]
    return features, labels


########################################################################################################################


def determine_material_value(material, true_results):
    """
    Performs pseudo experiment for the AMI where the performance value of the AMI selected material is looked up in the
    loaded data array

    :param material: int, index of the material chosen in the target values
    :param true_results: np.array(), `m` sized array containing the target values for the passed features
    :return: determined_value: float, the target value for the passed material index
    """
    determined_value = true_results[material, 0]
    return determined_value


########################################################################################################################

data_path = r'C:\Users\crh53\OneDrive\Desktop\PHD_Experiments\E2_AMI_James\Data\Scaled_HMOF_Data'
num_initial_samples = 100
max_iterations = 2000

X, y = data_loader(data_path)
n, d = X.shape

y_true = y.reshape(-1, 1)
y_experimental = np.full((n, 1), np.nan)  # column vector of determined material performances

top = np.argsort(y)[-100:]  # true top 100 to compare sample with
status = np.zeros((n, 1))  # column vector denoting status of material (0=untested, 1=testing, 2=tested)

n_tested = 0
initial_samples = np.random.randint(0, n, num_initial_samples)  # choose n values where each value is a material index
for sample in initial_samples:
    n_tested += 1
    status[sample] = 2
    y_experimental[sample] = determine_material_value(sample, y_true)  # determine material value and update

P = BOGP.prospector(X)

while n_tested < max_iterations:  # lets go!
    P.fit(y_experimental, status)
    ipick = P.pick_next(status)  # sample next point
    status[ipick, 0] = 1  # show that we are testing ipick
    y_experimental[ipick, 0] = determine_material_value(ipick, y_true)  # now lets get the score and update status
    status[ipick, 0] = 2
    n_tested = n_tested + 1  # count sample and print out current score
    print(n_tested)
    print(sum(1 for i in range(n) if i in top and status[i, 0] == 2))
