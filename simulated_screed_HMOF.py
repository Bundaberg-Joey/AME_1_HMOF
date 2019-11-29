#!/usr/bin/env python3

import argparse
import numpy as np
import BOGP


class SimulatedScreener(object):
    """Class which uses the AMI to perform simulated screening of materials from a dataset containing all features
    and target values for the entries.

    The SimulatedScreener must first be initialised by the user with the data location, the number of initial samples
    for the AMI to take, and the maximum number of iterations the screening should be ran till
    """

    def __init__(self, data_path, num_initial_samples, max_iterations):
        self.data_path = data_path
        self.num_initial_samples = num_initial_samples
        self.max_iterations = max_iterations
        self.n_tested = 0
        self.X = None
        self.n = None
        self.y = None
        self.y_true = None
        self.y_experimental = None
        self.status = None
        self.top_100 = None


    def simulation_initialisation(self):
        """
        Sets up all attributes required for running the AMI Simulated Screening
        :return: N/A updates internal parameters
        """
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

        def format_target_values(y, n):
            """
            For simulated screenings, AMI requires an experimental column of results it has determined itself and the
            "True" target values which it uses to evaluate chosen materials against. These must be in the correct
            matrix / vector shape.
            :return: (y_true, y_experimental), column vectors, [0] with all target values, [1] for determined values
            """
            y_true = y.reshape(-1, 1)  # column vector
            y_experimental = np.full((n, 1), np.nan)  # nan as values not yet determined on initialisation
            return y_true, y_experimental

        self.X, self.y = load_simulation_data(self.data_path)
        self.n = self.X.shape[0]
        self.y_true, self.y_experimental = format_target_values(self.y, self.n)
        self.top_100 = np.argsort(self.y)[-100:]
        self.status = np.zeros((self.n, 1))


    @staticmethod
    def _determine_material_value(material, true_results):
        """
        Performs pseudo experiment for the AMI where the performance value of the AMI selected material is looked up in
        the loaded data array
        :param material: int, index of the material chosen in the target values
        :param true_results: np.array(), `m` sized array containing the target values for the passed features
        :return: determined_value: float, the target value for the passed material index
        """
        determined_value = true_results[material, 0]  # 0 because column vector
        return determined_value


    def initial_random_samples(self):
        """
        Selects a number of random materials for the AMI to assess and performs pseudo experiments on all of them
        in order for the model to have initial data to work with
        :return: N/A updates internal parameters
        """
        initial_materials = np.random.randint(0, self.n, self.num_initial_samples)  # n random index values
        for material_index in initial_materials:
            self.status[material_index] = 2
            self.y_experimental[material_index] = self._determine_material_value(material_index, self.y_true)
            self.n_tested += 1


    def _user_updates(self):
        """
        Provides user updates on the status of the AMI screening. The current AMI iteration is provided along with the
        number of top 100 performing materials (determined from loaded dataset) also.
        """
        checked_materials = np.where(self.status[:, 0] == 2)[0]
        top_materials_found = sum(1 for i in range(self.n) if i in self.top_100 and i in checked_materials)
        print(F'AMI Iteration {self.n_tested}')
        print(F'{top_materials_found} out of 100 top materials found')


    def perform_screening(self, autonomous_model, verbose=True):
        """
        Performs the simulated screening on the loaded dataset using the passed model. For each iteration of the model:
        1) The model fits itself to target values it has learned through running experiments of selected materials
        2) Picks a new material to assess based on its features
        3) The status of the material is then updated (0=not yet assessed, 1=being assessed, 2=has been assessed)
        4) The target value of the material is then determined (index look up of the originally loaded dataset)

        Because the AMI requires a column vector be passed to it as `y_experimental` and `status` the [0] indexing
        seen is to support that functionality while also updating parameters on this side

        :param autonomous_model: The AMI object performing the screening of the materials being investigated
        :param verbose: Boolean, True sets the screener to provide user updates, False to silence them
        :return: N/A, updates internal parameters
        """
        while self.n_tested < self.max_iterations:

            autonomous_model.fit(self.y_experimental, self.status)
            ipick = autonomous_model.pick_next(self.status)  # sample next point
            self.status[ipick, 0] = 1  # show that we are testing ipick
            self.y_experimental[ipick, 0] = self._determine_material_value(ipick, self.y_true)
            self.status[ipick, 0] = 2
            self.n_tested += 1  # count sample and print out current score

            if verbose:
                self._user_updates()


if __name__ == '__main__':

    test_path = r'C:\Users\crh53\OneDrive\Desktop\PHD_Experiments\E2_AMI_James\Data\Scaled_HMOF_Data'
    # not moving file closer to preserve fidelity of the file generation

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', action='store', default=test_path, help='path to data file for screening')
    parser.add_argument('-i', '--initial_samples', action='store', default=100, help='# of random samples AMI takes')
    parser.add_argument('-m', '--max_iterations', action='store', default=2000, help='# of materials AMI will sample')
    args = parser.parse_args()

    experiment = SimulatedScreener(args.data_file, args.initial_samples, args.max_iterations)
    experiment.simulation_initialisation()
    ami = BOGP.prospector(experiment.X)
    experiment.initial_random_samples()
    experiment.perform_screening(ami)

# TODO : Re-write simulation_initialisation into it's own object class (will aid different file types later)
# TODO : Put simple tests in main to make sure data loaded ok (i.e. shape of X, y_true, status etc
