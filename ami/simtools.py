"""
Contains utility functions and classes for ami investigations.
Currently included are:
* Status
"""

import numpy as np

from ami import _checks


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
    Status object uses numpy arrays which require all entries to be the same type.
    Therefore label used should match the type of label used at initialisation.
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

        self.num = num
        self.start = start
        self.state = np.full(self.num, self.start)
        self.changelog = []

    def update(self, identifier, label):
        """Update the label of a particular experiment.
        Order of update is conserved in log.

        Notes
        -----
        Status object uses numpy arrays which require all entries to be the same type.
        Therefore label used should match the type of label used at initialisation.

        Parameters
        ----------
        identifier : int OR list of int
            The index(es) of the data point(s) to be updated.
            Note if multiple data points are updated simultaneously then this is reflected in the log.

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


# ----------------------------------------------------------------------------------------------------------------------


class Evaluator(object):
    """Used to assess how many top performes the search algorithm has found.

    Attributes
    ----------
    y : list / np.array(), shape(num_entries, )
        List or array of True target values.

    n : int > 0 (default = 100)
        The number of top materials to be considered by evaluator.

    _y_sorted : np.array(), shape(num_entries, )
        Indices of top values sorted from lowest to highest

    top_n : np.array(), shape(n, )
        Indices of the target values with the highest score.
        This can be inverted to be the lowest score by calling `invert_top`.

    Methods
    -------
    _determine_found(self, found) --> list indices which have been found that are present in `top_n`
    invert_top(self) --> Inverts the `top_n` indices from highest target values, to lowest target values.
    top_found_count(self, found) --> Returns the count of passed indices which have been found in the `top_n`.
    top_found_id(self, found) --> Returns the passed indices which are present in the `top_n`.

    Notes
    -----
    Determinatin of the top found will not consider the presence of duplicates in either user input or top_n.

    Updating the value of `n` with either the attribute or setter will also update the `top_n` attribute accordingly.
    By defulat however, it will return the indices of the target values with the highest scores.
    Therefore user will have to call `invert_top` if updating lowest `n`.
    """

    def __init__(self, y, n=100):
        """Target values sorted at initialisation to prevent duplication of sorting if user wishes to invert.
        Setters ensure that `n` is > 0 and type int.
        Also ensure `y` has no `nan` values and is not empty.

        Parameters
        ----------
        y : list / np.array(), shape(num_entries, )
            List or array of True target values.

        n : int > 0 (default = 100)
            The number of top materials to be considered by evaluator.
        """
        self.y = y
        self._y_sorted = np.argsort(self.y)
        self.n = n
        self.top_n = self._y_sorted[-self.n:]

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
            Values which are present in user input and `top_n`
        """
        found = np.asarray(found)
        are_top = found[np.isin(found, self.top_n)]
        return are_top

    def invert_top(self):
        """Sets `top_n` to be the indices of the lowest `n` target values.

        Returns
        -------
        None
        """
        self.top_n = self._y_sorted[:self.n]

    def top_found_count(self, found):
        """Returns the number of user inputs which match those in `top_n`.

        Parameters
        ----------
        found : list, shape(num_matches, )
            values which are present in user input and `top_n`

        Returns
        -------
        found : int
            Number of matches between user input and `top_n`.
        """
        return len(self._determine_found(found))

    def top_found_id(self, found):
        """Returns the identities of user inputs which match those in `top_n`.

        Parameters
        ----------
        found : list, shape(num_matches, )
            values which are present in user input and `top_n`

        Returns
        -------
        found : list, shape(num_matches, )
            Ids which match between user input and `top_n`.
        """
        return self._determine_found(found)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, array):
        """Ensure array is not empty and no `Nan` values present.
        Raises errors if so.
        """
        _checks.array_not_empty(array)
        _checks.nan_present(array)
        self._y = array

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        """Set value to positive integer. Raise error if incorrect argument type.
        Also re-updates `top_n` to be the highest `n` target values.
        """
        _checks.pos_int(value)
        self.top_n = self._y_sorted[-value:]
        self._n = value


# ----------------------------------------------------------------------------------------------------------------------


