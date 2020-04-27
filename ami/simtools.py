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
     >> found = np.array([[1, 5, 6, 7], [3, 1, 20, 10])
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
        self._top_n = top_n

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
        _checks.boolean(inverted)

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
        found = np.asarray(found)
        are_top = found[np.isin(found, self._top_n)]
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


