"""
Used for checking user assignemnt of values.
Functions ensure correct type and value of values are pased.
Used for setters, etc.
"""

import numpy as np


def pos_int(*args):
    """checks if valus is positive integer and returns if true.
    Otherwise raises relevant error.

    Parameters
    ----------
    args : iterable
        Positive integer or other user input.

    Returns
    -------
    None
    """
    for a in args:
        if not isinstance(a, int):
            raise TypeError('Value must be an integer')
        if a <= 0:
            raise ValueError('Value must be positive')


def any_float(*args):
    """checks if valus is float and returns if true.
    Otherwise raises relevant error.

    Parameters
    ----------
    args : iterable
        Float or other user input.

    Returns
    -------
    None
    """
    for a in args:
        if not isinstance(a, float):
            raise TypeError('Value must be a float')


def nan_present(*args):
    """Checks for the presence of `Nan` values in numpy arrays.
    Raises `ValueError` is present.

    Parameters
    ----------
    args : iterable
        np.array(), shape(num_entries, num_features)
        Array(s) to check for presence of nan

    Returns
    -------
    None
    """
    for a in args:
        if np.isnan(a).sum() > 0:
            raise ValueError('Nan values must not be present in array')


def array_not_empty(*args):
    """Checks if passed numpy array(s) are empty or not.
    Raises `ValueError` if empty.

    Notes
    -----
    Numpy array `.size` attribute is used instead of length to handle instances
    of nested empty lists i.e.:

    my_list = [[], [], []]
    my_array = np.array(my_list)
    len(my_list) --> 3
    my_array.size --> 0

    Parameters
    ----------
    args : iterable
        np.array(), shape(num_entries, num_features)
        Array(s) to check for presence of nan

    Returns
    -------
    None
    """
    for a in args:
        try:
            a = np.array(a)
            if a.size == 0:
                raise ValueError('Array is empty')
        except:
            raise ValueError('Input could not be converted to array')
            # bare exception used here incase value can't be array which should crash anyway.
