
"""
This file contains classes used to run the simulated screenings of the AMI either in series or parallel
"""

__author__ = 'Calum Hand'
__version__ = '1.1.2'


from datetime import datetime
import uuid

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
        self.file_uuid = str(uuid.uuid4())


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
            'file_uuid': self.file_uuid
        }

        output_name = F'ami_output_{self.file_uuid}.mat'
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

class SimulatedScreenerParallel(object):

    def __init__(self, model, data_params, test_cost, sim_budget, nthreads):
        self.model = model
        self.data_params = data_params  # obj, contains triaged data including `y_true`, `y_experimental` and `status`
        self.test_cost = test_cost  # float, cost incurred when `assessing` a candidate
        self.sim_budget = sim_budget  # float, total amount of resources which can be used for the screening
        self.nthreads = nthreads  # int, number of threads to work on
        self.history = []  # lists to store simulation history
        self.workers = [(0, 0)] * self.nthreads  # now follow controller, set up initial jobs separately
        self.finish_time = np.zeros(self.nthreads)  # times the workers will finish at, here is all zero


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


    def _screener_init(self):
        """
        starts by selecting a material and performing cheap and expensive test on it
        """
        subject = 0
        self.model.uu.remove(subject)  # selects from untested and performs experiments
        self.model.tt.append(subject)
        self.sim_budget -= self.test_cost  # update budget
        self.data_params.y_experimental[subject] = self.determine_material_value(subject, self.data_params.y_true)

    def _select_and_run_experiment(self, i):
        """
        Passed model selects a material to sample.
        If the material has not been tested before then a cheap test is run, otherwise run expensive.
        After each test, the budget is updated (contained within the model ?) and the worker finish time updated
        :param i: int, index of the worker to perform the task
        """
        ipick = self.model.pick()
        self.workers[i] = (ipick, 'y')
        self.model.ty.append(ipick)
        self.sim_budget -= self.test_cost
        self.finish_time[i] += np.random.uniform(self.test_cost, self.test_cost * 2)

    def _record_experiment(self, final):
        """
        After each experiment has been run, need to figure out the worker that will finish next.
        After each experiment, the model has to update its internal records of what has been tested and how.
        It then will update the history of the screening.
        Finally the index of the worker which has now finished is returned so that more work can be assigned.
        If the final parameter is `True` then there is no need to assign further work and so jobs are killed
        :param final: Boolean, indicates if on the final loop and should return anything or not
        :return: i: int, the index of the worker which is going to finish first
        """
        i = np.argmin(self.finish_time)  # get the worker which is closest to finishing
        idone = self.workers[i][0]

        self.model.ty.remove(idone)
        self.data_params.y_experimental[idone] = self.determine_material_value(idone, self.data_params.y_true)
        self.model.tu.remove(idone)
        self.model.tt.append(idone)
        self.history.append((idone, 'y'))

        if final:
            self.workers[i] = None
            self.finish_time[i] = np.inf
        else:
            return i

    def full_screen(self):
        """
        Performs the full automated screening with multiple workers.
        First each worker (determined by the number of threads) is assigned a material to investigate.
        After this initialisation, the screener alternates selecting and recording experiments.
        This proceeds until the budget is spent (all the while recording the history of the work).
        After the budget is spent s.t. no expensive tests can be run, the remaining jobs finish.
        :return: self.history: list, full accounting of what materials were sampled when and where
        """
        self._screener_init()  # initialise the model with a single expensive test

        for i in range(self.nthreads):  # at the start, give the workers a job to do each
            self._select_and_run_experiment(i)

        while self.sim_budget >= self.test_cost:  # spend budget till cant afford any more expensive tests
            i = self._record_experiment(final=False)
            self._select_and_run_experiment(i)

        for i in range(self.nthreads):  # finish up any remaining jobs and record their results
            self._record_experiment(final=True)

        return self.history


# TODO: update SimulatedScreenerParallel to work with `AME_1`
# TODO: update SimulatedScreenerParallel to accept y_true, y_experimental, STATUS from DataTriage
