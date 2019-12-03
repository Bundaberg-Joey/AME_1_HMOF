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

    def __init__(self):
        self.n = None
        self.y_true = None
        self.y_experimental = None
        self.status = None

    @staticmethod
    def load_simulation_data(data_path, data_delimiter=',', headers_present=1):
        """
        Loads the features and target variables for the AME to assess from a delimited file, assumed csv.
        The default loading removes the first row to allow headed files to be read and so should be specified if not.
        The delimited file is assumed to be structured with target values as the final right hand column.

        :param data_path: str, location of the the data file to be read
        :param data_delimiter: str, delimiter used in the file
        :param headers_present: int, Number of lines in data file head containing non numeric data, needs to be removed
        :return: features: np.array(), `m` by 'n' array which is the feature matrix of the data being modelled
        :return: labels: np.array(), `m` sized array containing the target values for the passed features
        """
        data_set = np.loadtxt(data_path, delimiter=data_delimiter, skiprows=headers_present)
        assert data_set.size > 0, 'Loaded data set was empty'
        features, labels = data_set[:, :-1], data_set[:, -1]
        return features, labels

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

    def prepare_simulation_data(self, y):
        """
        Updates all relevant object attributes with those determined from the loaded dataset. These attributes are then
        returned as a dictionary so that they can be further utilised as the basis for screening experiments on this
        loaded dataset.

        :param y: np.array(), `m` sized array containing the target values for the passed features
        :return: triaged_parameters: dict, export the technicians attributes to be used later by AMI
        """
        self.n = y.shape[0]
        self.y_true, self.y_experimental = self.format_target_values(y, self.n)
        self.status = np.zeros((self.n, 1))
        triaged_parameters = vars(self)
        return triaged_parameters


########################################################################################################################


class SimulatedScreener(object):
    """Class which uses an AMI model to perform simulated screening of materials from a dataset containing all features
    and target values for the entries.

    The simulated screener takes parameters about the data as input along with the maximum number of iterations
    that the model will run for.
    """
    def __init__(self, simulation_params, max_iterations):
        self.max_iterations = max_iterations
        self.n_tested = 0

        self.n = simulation_params['n']
        self.y_true = simulation_params['y_true']
        self.y_experimental = simulation_params['y_experimental']
        self.status = simulation_params['status']

        self.top_100 = np.argsort(self.y_true.ravel())[-100:]

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
        initial_materials = np.random.randint(0, self.n, num_initial_samples)  # n random index values
        for material_index in initial_materials:
            self.status[material_index] = 2
            self.y_experimental[material_index] = self.determine_material_value(material_index, self.y_true)
            self.n_tested += 1

    def user_updates(self):
        """
        Provides user updates on the status of the AMI screening. The current AMI iteration is provided along with the
        number of top 100 performing materials (determined from loaded dataset) also.
        """
        checked_materials = np.where(self.status[:, 0] == 2)[0]
        top_materials_found = sum(1 for i in range(self.n) if i in self.top_100 and i in checked_materials)
        print(F'AMI Iteration {self.n_tested}')
        print(F'{top_materials_found} out of 100 top materials found')

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

            model.fit(self.y_experimental, self.status)
            ipick = model.pick_next(self.status)  # sample next point
            self.status[ipick, 0] = 1  # show that we are testing ipick
            self.y_experimental[ipick, 0] = self.determine_material_value(ipick, self.y_true)
            self.status[ipick, 0] = 2
            self.n_tested += 1  # count sample and print out current score

            if verbose:
                self.user_updates()


########################################################################################################################
