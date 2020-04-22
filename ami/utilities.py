"""
Contains utility functions and classes for ami investigations.
Currently included are:
* Status
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
