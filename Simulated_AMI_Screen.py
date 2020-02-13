#!/usr/bin/env python3

import json
from uuid import uuid4
import os

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
    with open(F'json_output/{file_id}.json', 'w') as f:
        f.write(json.dumps(data, indent=4))


num_repetitions = 10
csv_files = [i for i in os.listdir('..') if '.csv' in i]

for data_file in csv_files:
    for rep in range(num_repetitions):

        sim_data = DataTriageCSV.load_from_path(data_file)  # load data and format

        sim_screen = SimulatedScreenerSerial(data_params=sim_data, max_iterations=501, sim_code='N/A')
        ami = BOGP.prospector(X=sim_data.X, acquisition_function='Thompson')
        # initialises the AMI model and the simulation screener with the triaged data
    
        sim_screen.initial_random_samples(num_initial_samples=500)  # sample 500 materials and then do killswitch
        z_mu, z_var = sim_screen.perform_screening(model=ami)

        ame_score = prediction_score(sim_data.y, z_mu)  # calc score needed for test
    
        columns = pd.read_csv(data_file).columns
        features, target = list(columns[:-1]), str(columns[-1])
    
        material, feat_code, bad_target = data_file.split('_')
    
        results_dict = {'features': features,
                        'target': target,
                        'ami_score': ame_score,
                        'data_set': data_file,
                        'feature_code': feat_code,
                        'material': material}
    
        save_data(results_dict)


