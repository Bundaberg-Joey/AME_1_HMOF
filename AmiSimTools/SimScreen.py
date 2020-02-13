"""
This file contains classes used to run the simulated screenings of the AMI either in series or parallel
"""

__author__ = 'Calum Hand'
__version__ = '2.1.2'

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


    def perform_screening(self, model):
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

        model.fit(self.data_params.y_experimental, self.data_params.status)
        z_mu, z_var = model.predict()
        return z_mu, z_var



########################################################################################################################

class SimulatedScreenerParallel(object):
    """Class which uses an AMI model to perform a parallel simulated screening of materials from a dataset
    containing all features and target values for the entries.
    The simulated screener takes a `data_params` object containing attributes calculated from the initial data
    used for the simulation. It's values are "composed" out of the object for use here
    """

    def __init__(self, data_params, test_cost, sim_budget, nthreads, num_init, min_samples):
        """
        :param data_params: DataTriage obj, contains triaged data including `y_true`, `y_experimental` and `status`
        :param test_cost: float, cost incurred when `assessing` a candidate
        :param sim_budget: float, total amount of resources which can be used for the screening
        :param nthreads: int, number of threads to work on
        :param num_init: int, number of initial random samples to take
        :param min_samples: int, number of materials to be assessed before AMI model can fit
        """
        self.data_params = data_params
        self.test_cost = test_cost
        self.sim_budget = sim_budget
        self.nthreads = nthreads
        self.num_init = num_init
        self.min_samples = min_samples

        assert self.num_init > self.nthreads, F'number of initial samples {self.num_init} < num threads {self.nthreads}'

        self.workers = [None] * self.nthreads
        self.finish_time = np.zeros(self.nthreads)
        self.queued = np.random.choice(self.data_params.n, self.num_init, replace=False)  # random material indices
        self.history = []
        self.model_fitted = False

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


    def _log_history(self, **kwargs):
        """
        Logs the history of the parallel screening.
        Continually updates by appending a dictionary to the history attribute which is eventually returned as a list.
        The list of dictionaries can then be readily converted into a pandas DataFrame after screening is complete.
        The start and end of experiments require different details saved, pandas allows for multiple different keywords
        allowing different keys to be passed depending on time of experiment.
        :param kwargs: dict, contains useful information about the simulated screening
        """
        self.history.append(kwargs)


    def _run_experiment(self, i, ipick, exp_note):
        """
        Passed model selects a material to sample.
        If the material has not been tested before then a cheap test is run, otherwise run expensive.
        After each test, the budget is updated (contained within the model ?) and the worker finish time updated
        :param i: int, index of the worker to perform the task
        :param ipick: int, index of the material to be assessed
        """
        self.workers[i] = ipick
        experiment_cost = np.random.uniform(self.test_cost, self.test_cost * 2)
        self.sim_budget -= experiment_cost

        start = self.finish_time[i]
        self.data_params.status[ipick] = 1  # update status
        self.finish_time[i] += experiment_cost

        self._log_history(note=exp_note, worker=i, candidate=ipick, time=start, exp_len=experiment_cost)


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
        idone = self.workers[i]

        experimental_value = self.determine_material_value(idone, self.data_params.y_true)
        self.data_params.y_experimental[idone] = experimental_value
        self.data_params.status[idone] = 2  # update status
        end = self.finish_time[i]

        self._log_history(note='end', worker=i, candidate=idone, time=end, exp_value=experimental_value)

        if final:
            self.workers[i] = None
            self.finish_time[i] = np.inf
        else:
            return i


    def _fit_if_safe(self, model):
        """
        If enough materials have been assessed as per user threshold, then allow the model to fit on the data
        obtained. If not enough then could potentially cause the model to crash on fitting which ruins the experiment.
        :param model: AMI model
        """
        complete_experiment = 2
        if sum(self.data_params.status == complete_experiment) >= self.min_samples:
            model.fit(self.data_params.y_experimental, self.data_params.status)
            self.model_fitted = True


    def _initial_materials(self):
        """
        Assigns each of the available workers/threads a random material from the initial queue.
        The worker then runs the material, and the queue is updated accordingly to remove the material.
        The queue is updated to ensure the same material isn't allocated multiple times.
        """

        for i in range(self.nthreads):
            ipick = self.queued[0]
            self.queued = np.delete(self.queued, 0)
            self._run_experiment(i, ipick, 'start initial sample')


    def perform_screening(self, model):
        """
        Performs the full automated screening with multiple workers.
        First each worker (determined by the number of threads) is assigned a material to investigate.
        After this initialisation, the screener alternates selecting and recording experiments.
        This proceeds until the budget is spent (all the while recording the history of the work).
        After the budget is spent s.t. no expensive tests can be run, the remaining jobs finish.
        :param model: The AMI object performing the screening of the materials being investigated
        :return: self.history: list, full accounting of what materials were sampled when and where
        """
        self._initial_materials()

        while self.sim_budget >= self.test_cost:  # spend budget till cant afford any more expensive tests

            i = self._record_experiment(final=False)

            if len(self.queued) > 0:  # if queued materials then sample, else let model sample
                ipick = self.queued[0]
                note = 'start initial sample'
                self.queued = np.delete(self.queued, 0)
            else:
                self._fit_if_safe(model)
                if self.model_fitted:
                    ipick = model.pick_next(self.data_params.status)  # fit model and then allow to pick
                    note = 'start ami sample'
                else:
                    ipick = np.random.choice([i for i in range(self.data_params.n) if self.data_params.status[i] == 0])
                    note = 'start ami rand sample'
            self._run_experiment(i, ipick, exp_note=note)

        for i in range(self.nthreads):  # finish up any remaining jobs and record their results
            self._record_experiment(final=True)

        return self.history
