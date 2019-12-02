#!/usr/bin/env python3

import numpy as np


class DataTriage(object):
    """
    The Triage takes a dataset which is to be assessed by the AMI and then performs the necessary pre-processing
    steps on the data set including: loading the data into numpy arrays, formatting the target values into the correct
    representation and other values.

    The Technician stores these parameters as attributes which are then exported as a dictionary which can then be
    accessed by other objects and loaded into their screenings.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.X = None
        self.n = None
        self.y = None
        self.y_true = None
        self.y_experimental = None
        self.status = None
        self.top_n = None


    @staticmethod
    def load_simulation_data(data_path):
        """
        Loads the features and target variables for the AME to assess. Currently set up as numpy array loading but
        can easily retrofit depending on final decided file type
        :return: features: np.array(), `m` by 'n' array which is the feature matrix of the data being modelled
        :return: labels: np.array(), `m` sized array containing the target values for the passed features
        """
        data_set = np.loadtxt(data_path)
        features, labels = data_set[:, :-1], data_set[:, -1]
        return features, labels

    @staticmethod
    def _format_target_values(y, n):
        """
        For simulated screenings, AMI requires an experimental column of results it has determined itself and the
        "True" target values which it uses to evaluate chosen materials against. These must be in the correct
        matrix / vector shape.
        :return: (y_true, y_experimental), column vectors, [0] with all target values, [1] for determined values
        """
        y_true = y.reshape(-1, 1)  # column vector
        y_experimental = np.full((n, 1), np.nan)  # nan as values not yet determined on initialisation
        return y_true, y_experimental

    def prepare_technical_data(self, top_n=100):
        """
        Updates all relevant object attributes with those determined from the loaded dataset. These attributes are then
        returned as a dictionary so that they can be further utilised as the basis for screening experiments on this
        loaded dataset
        :param top_n: int, the number of top samples to consider in the target values for scoring the AMI performance
        :return: triaged_parameters: dict, export the technicians attributes to be used later by AMI
        """
        self.X, self.y = self.load_simulation_data(self.data_path)
        self.n = self.X.shape[0]
        self.y_true, self.y_experimental = self._format_target_values(self.y, self.n)
        self.top_n = np.argsort(self.y)[-top_n:]
        self.status = np.zeros((self.n, 1))
        triaged_parameters = vars(self)
        return triaged_parameters


#test_path = r'C:\Users\crh53\OneDrive\Desktop\PHD_Experiments\E2_AMI_James\Data\Scaled_HMOF_Data'
#abe = DataTriage(test_path)
#parameters = abe.prepare_technical_data()
#print(parameters.keys())


