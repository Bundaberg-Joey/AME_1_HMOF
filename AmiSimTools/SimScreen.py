
"""
This file contains classes used to run the simulated screenings of the AMI either in series or parallel
"""

__author__ = 'Calum Hand'
__version__ = '1.0.0'


from datetime import datetime

import numpy as np
from scipy.io import savemat


class SimulatedScreenerSerial(object):
    """Class which uses an AMI model to perform simulated screening of materials from a dataset containing all features
    and target values for the entries.

    The simulated screener takes a `data_params` object containing attributes calculated from the initial data
    used for the simulation. It's values are "composed" out of the object for use here
    """

    def __init__(self, data_params, max_iterations, sim_code='N/A'):
        self.max_iterations = max_iterations
        self.data_params = data_params  # compose from passed object
        self.sim_code = sim_code
        self.n_tested = 0
        self.top_100_found = []
        self.sim_start = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.test_order = []


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
            self.test_order.append(material_index)


    def user_updates(self, display=True):
        """
        Provides user updates on the status of the AMI screening. The current AMI iteration is provided along with the
        number of top 100 performing materials (determined from loaded dataset) also.

        :param display: Boolean, states if the output should be written to the screen or not during screening
        """
        checked_materials = np.where(self.data_params.status[:, 0] == 2)[0]
        top_100 = self.data_params.top_100
        self.top_100_found = [i for i in range(self.data_params.n) if i in top_100 and i in checked_materials]
        if display:
            print(F'AMI Iteration {self.n_tested}')
            print(F'{len(self.top_100_found)} out of 100 top materials found')


    def simulation_output(self):
        """
        Saves the status of the simulator object and chosen attributes of the data object.
        Currently saved to MatLab file for convenience but ideally will step up to hdf5 in future

        :return: N/a file output
        """
        screen_output = {
            'sim_code': self.sim_code,
            'max_iterations': self.max_iterations,
            'n_tested': self.n_tested,
            'test_order': self.test_order,
            'top_100_found': self.top_100_found,
            'sim_start': self.sim_start,
            'status': self.data_params.status,
            'y_experimental': self.data_params.y_experimental
        }

        output_name = 'ami_output_' + self.sim_start + '.mat'
        savemat(output_name, screen_output)


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
            """ this next line is the 'experiment happening' """
            material_value = self.determine_material_value(ipick, self.data_params.y_true)
            self.data_params.y_experimental[ipick, 0] = material_value
            self.data_params.status[ipick, 0] = 2
            self.n_tested += 1  # count sample and print out current score
            self.test_order.append(ipick)  # update the order of materials sampled

            self.user_updates(display=verbose)
            self.simulation_output()


########################################################################################################################

class SimulatedScreenerParallel(SimulatedScreenerSerial):
    pass  # currently just here as placeholder for when the parallel class is properly developed
