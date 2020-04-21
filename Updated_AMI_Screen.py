#!/usr/bin/env python3

import argparse

import numpy as np

from AmiSimTools.DataTriage import DataTriageCSV

from ami.model import Prospector
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

    ami = Prospector(X=X, updates_per_big_fit=10)
    # screning ---------------------------------------------------------------------------------------------------------

    while n_tested < args.max_iterations:

        tested, untested = status.tested(), status.untested()
        y_tested = y_exp[tested]
        ami.fit(y_tested, tested=tested, untested=untested)

        posterior = ami.sample_posterior(n_repeats=1)  # thompson sampling
        a = alpha.thompson(posterior)
        ipick = utilities.select_max_alpha(untested=untested, alpha=a)

        y_exp[ipick] = y_true[ipick]  # update experimental value with true value
        status.update(ipick, 2)

        sampled = status.tested()
        top_sampled = sum((1 for i in sampled if i in top_100))
        print(F'({n_tested}/{args.max_iterations}) : {top_sampled} of top 100 sampled')

        n_tested += 1
