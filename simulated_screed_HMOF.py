#!/usr/bin/env python3

import argparse

import BOGP
from AMI_Simulations import DataTriage, SimulatedScreener


if __name__ == '__main__':

    data_location = r'C:\Users\crh53\OneDrive\Desktop\PHD_Experiments\E2_AMI_James\Data\Scaled_HMOF_Data.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', action='store', default=data_location, help='path to data file')
    parser.add_argument('-i', '--initial_samples', action='store', default=100, help='# of random samples AMI takes')
    parser.add_argument('-m', '--max_iterations', action='store', default=2000, help='# of materials AMI will sample')
    args = parser.parse_args()

    dt = DataTriage()
    dt.load_simulation_data(data_path=args.data_file, data_delimiter=',', headers_present=1, output=False)
    sim_params = dt.prepare_simulation_data()

    sim_screen = SimulatedScreener(simulation_params=sim_params, max_iterations=args.max_iterations)
    sim_screen.initial_random_samples(num_initial_samples=args.initial_samples)

    ami = BOGP.prospector(X=sim_screen.X)
    sim_screen.perform_screening(model=ami, verbose=True)

# TODO : Put simple tests in main to make sure data loaded ok (i.e. shape of X, y_true, status etc
