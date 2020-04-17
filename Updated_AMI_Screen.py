#!/usr/bin/env python3

import argparse

import numpy as np

from AmiSimTools.DataTriage import DataTriageCSV
from AmiSimTools.SimScreen import SimulatedScreenerSerial

from ami.model import Prospector
from ami import alpha, utilities


if __name__ == '__main__':

    data_location = r'C:\Users\local-user\OneDrive\Desktop\PHD_Experiments\E3_AMI_finalisation\Data\E3_2\csvfiles' \
                    r'\COF_pct_deliverablecapacityvSTPv.csv '

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', action='store', default=data_location, help='path to data file')
    parser.add_argument('-i', '--initial_samples', action='store', type=int, default=50, help='# of random samples AMI takes')
    parser.add_argument('-m', '--max_iterations', action='store', type=int, default=300, help='# of materials AMI will sample')
    parser.add_argument('-a', '--acquisition', action='store', type=str, default='greedy_tau')
    args = parser.parse_args()

    # use old code to quickly get the various initialisation stuff dones for data loading etc
    data = DataTriageCSV.load_from_path(args.data_file)
    sim_screen = SimulatedScreenerSerial(data_params=data, max_iterations=args.max_iterations, sim_code='testing')
    sim_screen.initial_random_samples(num_initial_samples=args.initial_samples)  # do the random sampling and all that
    n_tested = args.initial_samples
    sample_method = args.acquisition

    X, y_true, y_exp, status, top_100 = data.X, data.y_true, data.y_experimental, data.status, data.top_100  # unpack

    # updates ---------------------------------------------------------------------------------------------------------
    ami = Prospector(X)
    while n_tested < args.max_iterations:

        ami.fit(y_exp, status)

        # --> Thompson
        if sample_method == 'thompson':
            posterior = ami.sample_posterior(n_repeats=1)
            a = alpha.thompson(posterior)

        # --> Greedy N
        elif sample_method == 'greedy_n':
            N = 100
            posterior = ami.sample_posterior(n_repeats=100)
            a = alpha.greedy_n(posterior, n=N)

        # --> EI
        elif sample_method == 'ei':
            mu_pred, var_pred = ami.predict()
            a = alpha.expected_improvement(mu_pred, var_pred, ami.y_max)

        # --> Greedy Tau
        elif sample_method == 'greedy_tau':
            N = 100
            posterior = ami.sample_posterior(n_repeats=10)
            tau = utilities.estimate_tau(posterior, n=N)
            mu_pred, var_pred = ami.predict()
            a = alpha.greedy_tau(mu_pred, var_pred, tau)

        # --> Random
        elif sample_method == 'random':
            a = alpha.random(len(X))

        else:
            print('No valid sampling method selected')
            break

        untested = np.where(status == 0)[0]
        ipick = utilities.select_max_alpha(untested=untested, alpha=a)

        y_exp[ipick, 0] = y_true[ipick, 0]  # update experimental value with true value
        status[ipick, 0] = 2  # update status

        sampled = np.where(status != 0)[0]
        top_sampled = sum((1 for i in sampled if i in top_100))
        print(F'({n_tested}/{args.max_iterations}) : {top_sampled} of top 100 sampled')

        n_tested += 1
