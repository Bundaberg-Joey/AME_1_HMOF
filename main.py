#!/usr/bin/env python3

import argparse
import os
from uuid import uuid4
import pickle

import numpy as np

from ami import alpha, simtools
from ami.model import Prospector
from ami.data import DataTriageCSV


if __name__ == '__main__':

    data_location = r'C:\Users\local-user\OneDrive\Desktop\PHD_Experiments\E3_AMI_finalisation\Data\E3_2\csvfiles' \
                    r'\COF_pct_deliverablecapacityvSTPv.csv '

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', action='store', default=data_location, help='path to data file')
    parser.add_argument('-i', '--initial_samples', action='store', type=int, default=50, help='# of random samples AMI takes')
    parser.add_argument('-m', '--max_iterations', action='store', type=int, default=1000, help='# of materials AMI will sample')
    parser.add_argument('-t', '--top_n', action='store', type=int, default=100, help='# of top materials to benchmark')
    parser.add_argument('-a', '--acquis', action='store', type=str, default='thompson', help='AMI acquis func')
    parser.add_argument('-s', '--save', action='store', type=str, default='pickle_output', help='location to save output')
    args = parser.parse_args()

    N = 100  # for greedy sampling
    acquisition = args.acquis.lower()

    n_tested = args.initial_samples

    # setting up -------------------------------------------------------------------------------------------------------
    data = DataTriageCSV.load_from_path(args.data_file)
    X, y_true, y_exp = data.X, data.y_true, data.y_experimental  # unpack

    rate_eval = simtools.Evaluator.from_unordered(y_true, args.top_n)
    status = simtools.Status(len(X), 0)

    # update status and experimental arrays with random samples
    random_samples = np.random.choice(len(y_true), n_tested, replace=False)
    for sample in random_samples:
        y_exp[sample] = y_true[sample]
        status.update(sample, 2)

    model = Prospector(X=X)
    train_filter = simtools.TrainingFilter(nmax=400, ntop=100, nrecent=100)

    # screening --------------------------------------------------------------------------------------------------------
    updates_per_big_fit = 10

    for prospector_iteration in range(n_tested, args.max_iterations):

        tested, untested = status.tested(), status.untested()
        y_tested = y_exp[tested]

        if prospector_iteration % updates_per_big_fit == 0:
            print('fitting hyperparameters')
            train, ytrain = train_filter.select_training_points(tested, y_tested)
            model.update_model_parameters(untested, train, ytrain)

        model.fit_posterior(y_tested, tested)

        if acquisition == 'thompson':
            posterior = model.sample_posterior(n_repeats=1)  # thompson sampling
            a = alpha.thompson(posterior)

        elif acquisition == 'greedy_n':
            posterior = model.sample_posterior(n_repeats=N)  # James implementation for greedy N
            a = alpha.greedy_n(posterior, N)

        elif acquisition == 'ei':
            mu, var = model.predict(return_variance=True)
            y_max = max(y_exp)
            a = alpha.expected_improvement(mu, var, y_max)

        elif acquisition == 'greedy_tau':
            posterior = model.sample_posterior(n_repeats=10)  # james implementation
            tau = alpha.estimate_tau(posterior, n=N)
            mu, var = model.predict(return_variance=True)
            a = alpha.greedy_tau(mu, var, tau)

        else:
            print('random sampling')
            a = alpha.random(len(y_exp))

        ipick = alpha.select_max_alpha(untested=untested, alpha=a)

        y_exp[ipick] = y_true[ipick]  # update experimental value with true value
        status.update(ipick, 2)

        sampled = status.tested()
        top_sampled = rate_eval.top_found_count(sampled)
        print(F'({prospector_iteration}/{args.max_iterations}) : {top_sampled} of top {len(rate_eval)} sampled')


# ----------------------------------------------------------------------------------------------------------------------
    # save ouput

    log = status.changelog
    top = rate_eval._top_n
    sim_data = vars(args)

    sim_data['log'] = log
    sim_data['top'] = top

    file_name = os.path.join(args.save, uuid4().hex+'.pkl')

    with open(file_name, 'wb') as f:
        pickle.dump(sim_data, f)


