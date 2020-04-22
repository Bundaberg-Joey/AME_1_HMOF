#!/usr/bin/env python3

import argparse

import numpy as np

from AmiSimTools.DataTriage import DataTriageCSV

from ami.model import Prospector, FrugalTrainer
from ami import alpha, utilities


if __name__ == '__main__':

    data_location = r'C:\Users\local-user\OneDrive\Desktop\PHD_Experiments\E3_AMI_finalisation\Data\E3_2\csvfiles' \
                    r'\COF_pct_deliverablecapacityvSTPv.csv '

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', action='store', default=data_location, help='path to data file')
    parser.add_argument('-i', '--initial_samples', action='store', type=int, default=50, help='# of random samples AMI takes')
    parser.add_argument('-m', '--max_iterations', action='store', type=int, default=120, help='# of materials AMI will sample')
    args = parser.parse_args()

    n_tested = args.initial_samples

    # setting up -------------------------------------------------------------------------------------------------------
    data = DataTriageCSV.load_from_path(args.data_file)
    X, y_true, y_exp, top_100 = data.X, data.y_true, data.y_experimental, data.top_100  # unpack
    status = utilities.Status(len(X), 0)

    # update status and experimental arrays with random samples
    random_samples = np.random.choice(len(y_true), n_tested, replace=False)
    for sample in random_samples:
        y_exp[sample] = y_true[sample]
        status.update(sample, 2)

    model = Prospector(X=X)
    ft = FrugalTrainer(nmax=80, ntop=20, nrecent=20)
    # screning ---------------------------------------------------------------------------------------------------------
    updates_per_big_fit = 10

    while n_tested < args.max_iterations:

        tested, untested = status.tested(), status.untested()
        y_tested = y_exp[tested]

        if n_tested % updates_per_big_fit == 0:
            print('fitting hyperparameters')
            train, ytrain = ft.select_training_points(tested, y_tested)
            model.update_model_parameters(untested, train, ytrain)

        model.fit_posterior(y_tested, tested)

        posterior = model.sample_posterior(n_repeats=1)  # thompson sampling
        a = alpha.thompson(posterior)
        ipick = alpha.select_max_alpha(untested=untested, alpha=a)

        y_exp[ipick] = y_true[ipick]  # update experimental value with true value
        status.update(ipick, 2)

        sampled = status.tested()
        top_sampled = sum((1 for i in sampled if i in top_100))
        print(F'({n_tested}/{args.max_iterations}) : {top_sampled} of top 100 sampled')

        n_tested += 1
