#!/usr/bin/env python3

import argparse

import BOGP

from AmiSimTools.DataTriage import DataTriageCSV
from AmiSimTools.SimScreen import SimulatedScreenerSerial


if __name__ == '__main__':

    data_location = r'C:\Users\crh53\OneDrive\Desktop\PHD_Experiments\E2_AMI_James\Data\Scaled_HMOF_Data.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', action='store', default=data_location, help='path to data file')
    parser.add_argument('-i', '--initial_samples', action='store', type=int, default=100, help='# of random samples AMI takes')
    parser.add_argument('-m', '--max_iterations', action='store', type=int, default=2000, help='# of materials AMI will sample')
    parser.add_argument('-s', '--sim_code', action='store', type=str, default='N/A', help='Reference code for simulation')
    parser.add_argument('-a', '--acquisition', action='store', type=str, default='Thompson', help='AMI acquisition func')
    args = parser.parse_args()

    sim_data = DataTriageCSV.load_from_path(args.data_file)
    # loads data from csv file and then determines `status` array and other parameters as dict needed for the screening

    sim_screen = SimulatedScreenerSerial(data_params=sim_data, max_iterations=args.max_iterations, sim_code=args.sim_code)
    ami = BOGP.prospector(X=sim_data.X, acquisition_function=args.acquisition)
    # initialises the AMI model and the simulation screener with the triaged data

    sim_screen.initial_random_samples(num_initial_samples=args.initial_samples)
    sim_screen.perform_screening(model=ami, verbose=True)
    # performs the screening of the loaded data set using the ami model after picking a number of initial samples

