#!/usr/bin/env python3

import argparse
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import BOGP
from AmiSimTools import DataTriage, SimulatedScreener


def main(seed, simulator, model, num_initial_samples, verbose):
    """
    Allows the simulated screenings to be conducted in parallel, allowing for use on HPC systems.
    The random state of each iteration is set by the passed seed and is updated in the screener meta data.

    :param seed: numpy.int, integer value used to set the state of the random process for each parallel simulation
    :param simulator: SimulatedScreener, object which performs the simulations with the model and updates meta data
    :param model: prospector, AMI mathematical workhorse which is performing the material sampling and analysis
    :param num_initial_samples: int, number of initial samples for the simulator to take before passing to AMI
    :param verbose: boolean, if True, print output for user, if False then print nothing
    :return: NoneType
    """
    np.random.RandomState(seed=seed)
    simulator.initial_random_samples(num_initial_samples=num_initial_samples)
    simulator.perform_screening(model=model, verbose=verbose)


if __name__ == '__main__':

    data_location = r'C:\Users\crh53\OneDrive\Desktop\PHD_Experiments\E2_AMI_James\Data\Scaled_HMOF_Data.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', action='store', default=data_location, help='path to data file')
    parser.add_argument('-i', '--initial_samples', action='store', default=100, help='# of random samples AMI takes')
    parser.add_argument('-m', '--max_iterations', action='store', default=2000, help='# of materials AMI will sample')
    parser.add_argument('-r', '--repetitions', action='store', default=0, help='# of repetitions to run per core')
    args = parser.parse_args()

    dt = DataTriage()
    X, y = dt.load_simulation_data(data_path=args.data_file, data_delimiter=',', headers_present=1)
    sim_params = dt.prepare_simulation_data(y=y)
    # loads data from csv file and then determines `status` array and other parameters as dict needed for the screening

    sim_screen = SimulatedScreener(simulation_params=sim_params, max_iterations=args.max_iterations)
    ami = BOGP.prospector(X=X)
    # initialises the AMI model and the simulation screener with the previously exported dict

    n_cpus = multiprocessing.cpu_count()
    num_scheduled_simulations = int(args.repetitions) * n_cpus if args.repetitions else 1
    simulation_seeds = np.random.choice(2000000, num_scheduled_simulations, replace=False)  # sample without replacement
    # determine the number of parallel process to run, if user flag is 0, only 1 will run

    worker = partial(main, simulator=sim_screen, model=ami, num_initial_samples=args.initial_samples, verbose=True)

    with ProcessPoolExecutor(max_workers=n_cpus) as pool:
        for res in pool.map(worker, simulation_seeds):
            pass
    # perform multi thread processing of numerous AMI runs

# TODO : add `update_metadata()` method to the SimulatedScreener class
