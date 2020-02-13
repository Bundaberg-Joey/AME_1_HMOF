#!/usr/bin/env python3

import argparse
import json
from uuid import uuid4

import numpy as np
import pandas as pd

import BOGP
from AmiSimTools.DataTriage import DataTriageCSV
from AmiSimTools.SimScreen import SimulatedScreenerSerial


def prediction_score(y, z):
    """
    calculate difference between ame predictions and true scores
    :param y: true scores
    :param z: predicted scores
    :return: difference
    """
    y_bar = y.mean()
    score = np.sqrt(np.sum(np.power(y-z, 2)) / np.sum(np.power(y-y_bar, 2)))
    return score


def save_data(data):
    """
    save data to json file, with uuid 4 named output file
    :param data: dict
    """
    file_id = str(uuid4())
    with open(F'{file_id}.json', 'w') as f:
        f.write(json.dumps(data, indent=4))



if __name__ == '__main__':

    data_location = r'C:\Users\crh53\OneDrive\Desktop\PHD_Experiments\E2_AMI_James\Data\Scaled_HMOF_Data.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', action='store', default=data_location, help='path to data file')
    parser.add_argument('-i', '--initial_samples', action='store', type=int, default=500, help='# of random samples AMI takes')
    parser.add_argument('-a', '--acquisition', action='store', type=str, default='Thompson', help='AMI acquisition func')
    args = parser.parse_args()

    sim_data = DataTriageCSV.load_from_path(args.data_file)
    # loads data from csv file and then determines `status` array and other parameters as dict needed for the screening

    sim_screen = SimulatedScreenerSerial(data_params=sim_data, max_iterations=501, sim_code='N/A')
    ami = BOGP.prospector(X=sim_data.X, acquisition_function=args.acquisition)
    # initialises the AMI model and the simulation screener with the triaged data

    sim_screen.initial_random_samples(num_initial_samples=args.initial_samples)
    z_mu, z_var = sim_screen.perform_screening(model=ami)
    ame_score = prediction_score(sim_data.y, z_mu)  # calc score needed for test

    columns = pd.read_csv(args.data_file).columns
    features, target = list(columns[:-1]), str(columns[-1])

    material, feat_code, bad_target = args.data_file.split('_')

    results_dict = {'features': features,
                    'target': target,
                    'ami_score': ame_score,
                    'data_set': args.data_file,
                    'feature_code': feat_code,
                    'material': material}

    save_data(results_dict)


