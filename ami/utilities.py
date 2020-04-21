"""
Contains utility functions and classes for ami investigations.
Currently included are:
* Status
* select max alpha
* estimate tau
"""

import numpy as np


class Status(object):
    """Object for tracking which experiments have been conducted on each data point.

    Attributes
    ----------
    num : int
        Number of entries to create a status for.

    start : {str, int, float} (default = 0)
        Value used to denote the starting state of all entries.
        Used to check for (non)tested entries in `tested` and `untested` methods.

    state : np.array(), shape(num, )
        Array which contains the status of each data point.
        Each data point has it's own entry which can be updated by the user to any value.

    changelog : list[(identifier, label)]
        List containing log of materials updated and the label updated to.


    Methods
    -------
    update(self, identifier, label) --> updates `self.state`
    tested(self) --> returns array of tested data points
    untested(self) --> returns array of untested data points

    Notes
    -----
    To allow flexibility, users can update the contents of status to be whatever they wish.
    This includes the intialisation value to prevent constraining the user.
    It is advised however that the user adheres to their own set up as none will be enforced here.
    """

    def __init__(self, num, start=0):
        """
        Parameters
        ----------
        num : int
            Number of entries to create a status for.

        start : {str, int, float} (default = 0)
            Value used to denote the starting state of all entries.
            Used to check for (non)tested entries in `tested` and `untested` methods.
        """
        self.num = num
        self.start = start
        self.state = np.full(self.num, self.start)
        self.changelog = []

    def update(self, identifier, label):
        """Update the label of a particular experiment.
        Order of update is conserved in log.

        Parameters
        ----------
        identifier : int
            The index of the data point to be updated

        label : {str, int, float}
            Any label which the user wishes to use to update an experiment with

        Returns
        -------
        None
        """
        self.state[identifier] = label
        self.changelog.append((identifier, label))

    def tested(self):
        """Provides indices of entries which have been tested.
        Compares the entries against `start` from initialisation and returns those that don't match.

        Returns
        -------
        tested : list, shape(<num matches>, )
            List of indices which do not match the `default` value
        """
        tested = np.where(self.state != self.start)[0]
        return list(tested)

    def untested(self):
        """Provides indices of entries which have not been tested.
        Compares the entries against `default` from initialisation and returns those that match.

        Returns
        -------
        tested : list shape(<num matches>, )
            List of indices which match the `default` value
        """
        untested = np.where(self.state == self.start)[0]
        return list(untested)


# ----------------------------------------------------------------------------------------------------------------------


class FrugalTrainer(object):
    """Determines ideal training points to consider to facilitate sparse fitting of the Prospector.

    Attributes
    ----------
    nmax : int
        The highest number of data points to be allowed in the training set.

    ntop : int
        The number of`top` data points to be included in the training set.

    nrecent : int
        The `n` most recently sampled data points to include in training data.


    Methods
    -------
    select_training_points(self, tested_indices, test_results) --> determines training data.

    Notes
    -----
    As the prospector works by considering nystrom inducing points rather than a full `n` by `n`
    covariance matrix, it is ideal to keep the size of the training data small.
    Here, the training set is contained to an upper limit of `nmax`, which is either the data
    acquired up till now or a more frugal selection.
    The frugal selection combines the `ntop` highest value data points, the `nrecent` training
    points, with the remainder being random data points which have already been tested.
    """

    def __init__(self, nmax=400, ntop=100, nrecent=100):
        """
        Parameters
        ----------
        nmax : int
            The highest number of data points to be allowed in the training set.

        ntop : int
            The number of`top` data points to be included in the training set.

        nrecent : int
            The `n` most recently sampled data points to include in training data.
        """
        self.nmax = nmax
        self.ntop = ntop
        self.nrecent = nrecent

    def select_training_points(self, tested_indices, test_results):
        """Selects training points to use, given imposed constraints on initialisation.
        If the number of tested points is less than the user maximum, then just returns passed arrays to user.

        Parameters
        ----------
        tested_indices : list / array, shape(num_tested_points, )
            Indices of data points in the feature matrix which have been tested.

        test_results : list / array, shape(num_tested_points, )
            Target values of data points which have been tested.

        Returns
        -------
        (train_indices, y_train) : tuple of np.array(), shape( 1<= `nmax`, )
            Indices of data points to use and their emperical target values.
            The size of the arrays are dependant on the conditional size.
        """

        tested_indices = np.array(tested_indices)
        y_tested = np.array(test_results)
        n_tested = len(y_tested)

        if n_tested > self.nmax:
            indices = np.arange(0, n_tested, 1)

            top_ind = np.argsort(y_tested)[-self.ntop:]
            recent_ind = indices[n_tested - self.nrecent:]
            top_recent_ind = np.unique(np.concatenate((top_ind, recent_ind)), axis=0)

            not_toprecent = indices[np.isin(indices, top_recent_ind, invert=True)]
            rand = np.random.choice(not_toprecent, self.nmax - len(top_recent_ind), replace=False)

            testedtrain = np.concatenate((top_recent_ind, rand))
            train_indices, y_train = tested_indices[testedtrain], y_tested[testedtrain]

        else:
            train_indices, y_train = tested_indices, test_results

        return train_indices, y_train


# ----------------------------------------------------------------------------------------------------------------------


def select_max_alpha(untested, alpha):
    """Given the alpha values for each member of the dataset and list of untested points,
    select the next point for investigation with the highest alpha value which has not yet been tested.

    Parameters
    ----------
    untested : list / np.array(), shape(num_not_tested, )
        Integer indices of the data points which have not yet been tested / sampled.

    alpha : np.array(), shape(num_database_entries, )
        Array containing the alpha values of each point within the data set.
        Alpha values are calculated using the acquisution functions in `ami.acquisition`.

    Returns
    -------
    ipick : int
        Integer index of the data point to be selected from the original data set.
    """
    max_untested_alpha = np.argmax(alpha[untested])
    pick = untested[max_untested_alpha]
    return pick


# ----------------------------------------------------------------------------------------------------------------------


def estimate_tau(posterior, n):
    """Used to calculate `tau` for `greedy_tau` sampling where `tau` is the threshold value for a data point to be
    in the top `n` data points.
    The threshold values for the top `n` data points for each sampling of the posterior are pooled and the returned
    tau value is the median of those thresholds.

    Note : As estimateing the `tau` value is not an inexpensive calculation, it is recommended to only update `tau`
    perdiodically rather than every model investigation iteration.

    Parameters
    ----------
    posterior : np.array(), shape(num_dataset_entries, num_posterior_samples)
        Posterior sampling of the dataset for each entry.
        Each entry (row) has `n` associated values / draws from the posterior.
        Dataset must have been sampled `n` times which can be achieved with `ami.samples(n)`.

    n : int
        The number of data points which the greedy `tau` algorithm is attempting to optimise for.
        i.e. if `n` = 100, calculates the number of times each data point appears in the top 100.

    Returns
    -------
    tau : float
        The median of the `n`th top data points from the sampled posterior.
    """
    a, b = posterior.shape

    taus = np.zeros(b)
    for i in range(b):
        taus[i] = np.sort(posterior[:, i])[-n]

    tau = np.median(taus)
    return tau
