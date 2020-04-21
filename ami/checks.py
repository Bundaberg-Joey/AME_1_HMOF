"""
Used for checking user assignemnt of values.
Functions ensure correct type and value of values are pased.
Used for setters, etc.
"""

import numpy as np

def pos_int(value):
    """checks if valus is positive integer and returns if true.
    Otherwise raises relevant error.

    Parameters
    ----------
    value : int
        positive integer or other user input.

    Returns
    -------
    value : int
        Value provided by user.
    """
    if not isinstance(value, int):
        raise TypeError('Value must be an integer')
    if value <= 0:
        raise ValueError('Value must be positive')
    return value


def any_float(value):
    """checks if valus is float and returns if true.
    Otherwise raises relevant error.

    Parameters
    ----------
    value : float
        Float or other user input.

    Returns
    -------
    value : int
        Value provided by user.
    """
    if not isinstance(value, float):
        raise TypeError('Value must be a float')
    return value


def nan_present(*args):
    """Checks for the presence of `Nan` values in numpy arrays.
    Raises `ValueError` is present.

    Parameters
    ----------
    args : np.array(), shape(num_entries, num_features)
        Array(s) to check for presence of nan

    Returns
    -------
    None
    """
    for a in args:
        if np.isnan(a).sum() > 0:
            raise ValueError('Nan values must not be present in array')


def array_empty(*args):
    """Checks if passed numpy array(s) are empty or not.
    Raises `ValueError` if empty.

    Parameters
    ----------
    args : np.array(), shape(num_entries, num_features)
        Array(s) to check for presence of nan

    Returns
    -------
    None
    """
    for a in args:
        if a.size == 0:
            raise ValueError('Array is empty')
