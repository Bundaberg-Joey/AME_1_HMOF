#!/usr/bin/env python3

"""
This module contains classes used to run simulated screenings with the AMI on already determined data sets.
"""

__author__ = 'Calum Hand'
__version__ = '2.2.0'

import warnings

import numpy as np
from scipy.io import loadmat


class DataTriage(object):
    """
    This Parent takes a dataset which is to be assessed by the AMI and then performs the necessary pre-processing
    steps on the data set including: loading the data into numpy arrays, formatting the target values into the correct
    representation and other values.

    Children of this class allow for multiple types of files to be loaded and used depending on user requirements
    """

    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.n = len(self.y)
        self.y_true, self.y_experimental = self.format_target_values(self.y, self.n)
        self.status = np.zeros((self.n, 1))
        self.top_100 = np.argsort(self.y)[-100:]


    @staticmethod
    def _load_dataset_from_path(data_path):
        """
        Method to be overloaded in further subclasses, loads data for the object

        :param data_path: str, path to data
        :return: x, y: (np.ndarray, np.ndarray)
        """
        return NotImplemented


    @classmethod
    def load_from_path(cls, data_path):
        """
        load data from the overloaded class method based on the location and then instantiate the object

        :param data_path: str, path to data
        :return: object, instantiated DataTriage object
        """
        X, y = cls._load_dataset_from_path(data_path)
        return cls(X, y)


    @staticmethod
    def format_target_values(y, n):
        """
        For simulated screenings, AMI requires an experimental column of results it has determined itself and the
        "True" target values which it uses to evaluate chosen materials against. These must be in the correct
        matrix / vector shape.

        :param y: np.array(), size `n` array containing the loaded target values
        :param n: int, the number of entries in the passed array `y`
        :return: (y_true, y_experimental), column vectors, [0] with all target values, [1] for determined values
        """
        y_true = y.reshape(-1, 1)  # column vector
        y_experimental = np.full((n, 1), np.nan)  # nan as values not yet determined on initialisation
        return y_true, y_experimental


class DataTriageCSV(DataTriage):
    """
    Child class which allows for the loading of data from a csv file
    """
    @staticmethod
    def _load_dataset_from_path(path: str) -> (np.ndarray, np.ndarray):
        """
        Loads the features and target variables for the AME to assess from a delimited file, assumed csv.
        The default loading removes the first row to allow headed files to be read and so should be specified if not.
        The delimited file is assumed to be structured with target values as the final right hand column.

        :param path: str, location of the csv data file to be read
        :return: features: np.array(), `m` by 'n' array which is the feature matrix of the data being modelled
        :return: targets: np.array(), `m` sized array containing the target values for the passed features
        """
        data_set = np.loadtxt(path, delimiter=",", skiprows=1)
        if data_set.size <= 0:
            warnings.warn('Loaded data set was empty')
        features, targets = data_set[:, :-1], data_set[:, -1]
        return features, targets


class DataTriageMatlab(DataTriage):
    """
    Child class which allows for the loading of data from a matlab file
    """
    @staticmethod
    def _load_dataset_from_path(path):
        """
        Loads the feature and target variables from a matlab file where the feature matrix and target array are in
        separate keys. The size of both the feature matrix and arrays are assessed and a warning is issued
        if the arrays are empty.

        :param path: str, location of the matlab file to be read
        :return: features: np.array(), `m` by 'n' array which is the feature matrix of the data being modelled
        :return: targets: np.array(), `m` sized array containing the target values for the passed features
        """
        data_set = loadmat(path, appendmat=False)
        feature_key, target_key = 'X', 'y'
        features, targets = data_set[feature_key], data_set[target_key]
        if features.size <= 0 or targets.size <= 0:
            warnings.warn('Loaded feature matrix or target array was empty')
        if features.shape[0] != targets.shape[0]:
            warnings.warn('The number of entries in the feature matrix and target array do not match')
        return features, targets


########################################################################################################################


class SimulatedScreener(object):
    """Class which uses an AMI model to perform simulated screening of materials from a dataset containing all features
    and target values for the entries.

    The simulated screener takes a `data_params` object containing attributes calculated from the initial data
    used for the simulation. It's values are "composed" out of the object for use here
    """

    def __init__(self, data_params, max_iterations):
        self.max_iterations = max_iterations
        self.data_params = data_params  # compose from passed object
        self.n_tested = 0
        self.top_100_found = []


    @staticmethod
    def determine_material_value(material, true_results):
        """
        Performs pseudo experiment for the AMI where the performance value of the AMI selected material is looked up in
        the loaded data array

        :param material: int, index of the material chosen in the target values
        :param true_results: np.array(), `m` sized array containing the target values for the passed features
        :return: determined_value: float, the target value for the passed material index
        """
        determined_value = true_results[material, 0]  # 0 because column vector indexing
        return determined_value


    def initial_random_samples(self, num_initial_samples):
        """
        Selects a number of random materials for the AMI to assess and performs pseudo experiments on all of them
        in order for the model to have initial data to work with

        :param num_initial_samples: int, number of data points to be sampled randomly from initial data
        :return: N/A updates internal parameters
        """
        initial_materials = np.random.choice(self.data_params.n, num_initial_samples, replace=False)
        # n random index values

        for material_index in initial_materials:
            material_value = self.determine_material_value(material_index, self.data_params.y_true)
            self.data_params.y_experimental[material_index] = material_value
            self.data_params.status[material_index] = 2
            self.n_tested += 1


    def user_updates(self):
        """
        Provides user updates on the status of the AMI screening. The current AMI iteration is provided along with the
        number of top 100 performing materials (determined from loaded dataset) also.
        """
        checked_materials = np.where(self.data_params.status[:, 0] == 2)[0]
        top_100 = self.data_params.top_100
        self.top_100_found = [i for i in range(self.data_params.n) if i in top_100 and i in checked_materials]
        print(F'AMI Iteration {self.n_tested}')
        print(F'{len(self.top_100_found)} out of 100 top materials found')

    def perform_screening(self, model, verbose=True):
        """
        Performs the simulated screening on the loaded dataset using the passed model. For each iteration of the model:
        1) The model fits itself to target values it has learned through running experiments of selected materials
        2) Picks a new material to assess based on its features
        3) The status of the material is then updated (0=not yet assessed, 1=being assessed, 2=has been assessed)
        4) The target value of the material is then determined (index look up of the originally loaded dataset)

        Because the AMI requires a column vector be passed to it as `y_experimental` and `status` the [0] indexing
        seen is to support that functionality while also updating parameters on this side

        :param model: The AMI object performing the screening of the materials being investigated
        :param verbose: Boolean, True sets the screener to provide user updates, False to silence them
        :return: N/A, updates internal parameters
        """
        while self.n_tested < self.max_iterations:

            model.fit(self.data_params.y_experimental, self.data_params.status)
            ipick = model.pick_next(self.data_params.status)  # sample next point
            self.data_params.status[ipick, 0] = 1  # show that we are testing ipick
            material_value = self.determine_material_value(ipick, self.data_params.y_true)
            self.data_params.y_experimental[ipick, 0] = material_value
            self.data_params.status[ipick, 0] = 2
            self.n_tested += 1  # count sample and print out current score

            if verbose:
                self.user_updates()

########################################################################################################################
