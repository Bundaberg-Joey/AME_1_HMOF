"""
Contains utility functions and classes for ami investigations.
Currently included are:
* Status
"""
import warnings

import numpy as np

from ami import _checks


class Status(object):
    """Object for tracking which experiments have been conducted on each data point.

    Attributes
    ----------
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
    get_status(self) --> returns satus array
    get_changelog(self) --> returns list of chronological updates to status
    len --> length of status array

    Notes
    -----
    Status object uses numpy arrays which require all entries to be the same type.
    Therefore label used should match the type of label used at initialisation.
    This is particularly important if storing status updates as strings as they must all be the same or less length.
    >> status = Status(3, start='a')
    >> status.update(0, 'bcd')
    >> status.get_status()
        np.array(['b', 'a', 'a'])
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
        _checks.pos_int(num)

        self.start = start
        self.state = np.full(num, self.start)
        self.changelog = []

    def update(self, identifier, label):
        """Update the label of a particular experiment.
        Order of update is conserved in log.
        The allowed update type can be viewed using `allowed_update_type(self)`

        Notes
        -----
        Status object uses numpy arrays which require all entries to be the same type.
        Therefore label used should match the type of label used at initialisation.
        When using strings, this means all strings should be same length, max length will be initial string length.

        Parameters
        ----------
        identifier : int
            The index of the status entries to be updated.

        label : self.state.dtype
            Label being used to indicate update in status.
            Type must match that of numpy array otherwise entries could be shortened.

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
        return tested

    def untested(self):
        """Provides indices of entries which have not been tested.
        Compares the entries against `default` from initialisation and returns those that match.

        Returns
        -------
        tested : list shape(<num matches>, )
            List of indices which match the `default` value
        """
        untested = np.where(self.state == self.start)[0]
        return untested

    def allowed_update_type(self):
        """Type of array used to store the status.

        Returns
        -------
        `state.dtype` : numpy.dtype
            Type of data it is reccomended user update the status with.
        """
        return self.state.dtype

    def get_status(self):
        """Provides user with status.

        Returns
        -------
        `self.state` : np.array(), shape(num_entries, )
            Status array tracking experiments.
        """
        return self.state

    def get_changelog(self):
        """Record of all updates made to the Status object, stored in chronological order.

        Returns
        -------
        `changelog` : list, shape(num_updates, )
            List of tuples (index, update) appended in chronological order
        """
        return self.changelog

    def __len__(self):
        """
        Returns
        -------
        length : int
            Length of status array.
        """
        return len(self.state)


# ----------------------------------------------------------------------------------------------------------------------


class Evaluator(object):
    """Used to assess how many top performes the search algorithm has found.

    Attributes
    ----------
    top_n : np.array(), shape(n, )
        Indices / labels of the target values considered to be the "top".

    Methods
    -------
    _determine_found(self, found) --> list indices which have been found that are present in `top_n`
    top_found_count(self, found) --> Returns the count of passed indices which have been found in the `top_n`.
    top_found_id(self, found) --> Returns the passed indices which are present in the `top_n`.
    get_top_n(self) --> return the `_top_n` array.
    len --> return length of `_top_n` array.

    Notes
    -----
    Evaluator will only consider duplicate values as being present once i.e.
    >> top = [1]
    >> ev = Evaluator(top)
    >> found = [1, 1, 2]
    >> ev.top_found_count(found)
        1
    >> ev.top_found_id(found)
        [1]

     It is possible to use the Evaluator to parse 2D arrays of `found` indices in `top_found` methods.
     However, if a 2D array (m \times n) is passed then only a 1D array will be returned with a label present
     if it exists **anywhere** within the passed 2D `found` grid i.e.
     >> top = [1, 3]
     >> ev = Evaluator(top)
     >> found = np.array([[1, 5, 6, 7], [3, 1, 20, 10]])
     >> ev.top_found_id(found)
        np.array([1, 3])
    """

    def __init__(self, top_n):
        """Checks that passed array is not empty and has no nan values present.
        
        Parameters
        ----------
        top_n : list / np.array(), shape(num_entries, )
            Indices / labels of target values considered `top`.
        """
        _checks.array_not_empty(top_n)
        _checks.nan_present(top_n)

        self._top_n = np.asarray(top_n)

    @classmethod
    def from_unordered(cls, y, n=100, inverted=False):
        """Factory method for when user wants a simple top `n` of a sorted array.
        `Inverted` option allows user to select if they wish for the highest of lowest `n` values.
        Useful for maximisation and minimisation problems.

        Parameters
        ----------
        y : list / np.array(), shape(num entries, )
            Array of target values.

        n : int (default = 100)
            The top `n` to be considered.

        inverted : bool (default = False)
            False if want top `n` largest values, True if want top `n` smallest values.

        Returns
        -------
        Evaluator(top)
            Instantiated `Evaluator` object from the passed data.
        """
        _checks.pos_int(n)
        _checks.array_not_empty(y)
        _checks.nan_present(y)
        _checks.are_type(bool, inverted)

        if n > len(y):
            warnings.warn(F'Specified argument `n` {n} is greater than pased array length {len(y)}', Warning)

        y_sorted = np.argsort(y)
        if not inverted:
            top_n = y_sorted[-n:]
        else:
            top_n = y_sorted[:n]
        return cls(top_n)

    def _determine_found(self, found):
        """Determine indices which are present in user input and `top_n`.

        Parameters
        ----------
        found : list, np.array(), shape(user_inpit, )
            Indices which the user are coparing.
            Can be any length so long is flat list / array.

        Returns
        -------
        are_top : np.array(), shape(num_matches, )
            Values which are present in user input and `top_n`.
        """
        are_top = self._top_n[np.isin(self._top_n, found)]
        return are_top

    def top_found_count(self, found):
        """Returns the number of user inputs which match those in `top_n`.
        Will also recount duplicates.

        Parameters
        ----------
        found : list, shape(num_matches, )
            Values which are present in user input and `top_n`.

        Returns
        -------
        found : int
            Number of matches between user input and `top_n`.
        """
        return len(self._determine_found(found))

    def top_found_id(self, found):
        """Returns the identities of user inputs which match those in `top_n`.
        Will also recount duplicates.

        Parameters
        ----------
        found : list, shape(num_matches, )
            values which are present in user input and `top_n`.

        Returns
        -------
        found : list, shape(num_matches, )
            Ids which match between user input and `top_n`.
        """
        return self._determine_found(found)

    def get_top_n(self):
        """Getter method for the `_top_n`.

        Returns
        -------
        _top_n : np.array(), shape(n_entries, )
            Array of top `n` indices .
        """
        return self._top_n

    def __len__(self):
        """
        Returns
        -------
        Length of array containing top values.
        """
        return len(self._top_n)


# ----------------------------------------------------------------------------------------------------------------------


class TrainingFilter(object):
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
        _checks.array_not_empty(tested_indices, test_results)
        _checks.nan_present(tested_indices, test_results)

        tested_indices = np.asarray(tested_indices)
        y_tested = np.asarray(test_results)
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

    @property
    def nmax(self):
        return self._nmax

    @nmax.setter
    def nmax(self, value):
        """Set value to positive integer. Raise error if incorrect argument type.
        """
        _checks.pos_int(value)
        self._nmax = value

    @property
    def ntop(self):
        return self._ntop

    @ntop.setter
    def ntop(self, value):
        """Set value to positive integer. Raise error if incorrect argument type.
        """
        _checks.pos_int(value)
        self._ntop = value

    @property
    def nrecent(self):
        return self._nrecent

    @nrecent.setter
    def nrecent(self, value):
        """Set value to positive integer. Raise error if incorrect argument type.
        """
        _checks.pos_int(value)
        self._nrecent = value
