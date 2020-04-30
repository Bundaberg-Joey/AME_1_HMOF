"""
Used for checking user assignemnt of values.
Functions ensure correct type and value of values are pased.
Used for setters, etc.
"""

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------


def are_type(types, *args):
    """Checks if passed args belong to specified type(s).
    Raises ValueError if singular rg not a member of specified types.

    Parameters
    ----------
    types : type OR tuple of types if multiple type checks
        Python types to check passed args against.
        Can pass either single type or tuple of multiple types.

    args : iterable
        Objects to be type checked.

    Returns
    -------
    None
    """
    for a in args:
        if not isinstance(a, types):
            raise TypeError(F'Passed arg must be member of {types} but {type(a)} was passed.')

# TODO : add string check to datatriage


# ----------------------------------------------------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------------------------------------------------


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

    This results in known limitation of non array / list types not raising an error.
    This also does not guard against partially empty arrays / lists (i.e. empty rows or columns)

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
            a = np.asarray(a)
            if a.size == 0:
                raise ValueError('Array is empty')
        except:
            raise ValueError('Input could not be converted to array')
            # bare exception used here incase value can't be array which should crash anyway.


# ----------------------------------------------------------------------------------------------------------------------


def same_shape(arrays):
    """Checks if passes arrays / lists / objects are the same shape.

    Notes
    -----
    Only really valid for lists or arrays since dictionaries will return shape() and other
    objects will have shape (1, )

    Parameters
    ----------
    arrays : iterable
        List or Array or tuple of objects to compare lengths of.

    Returns
    -------
    None
    """
    shapes = set((np.asarray(a).shape for a in arrays))
    if len(shapes) > 1:
        raise ValueError('Length mismatch, arrays do not have same shape')


# ----------------------------------------------------------------------------------------------------------------------


def same_type(comparison_type, *args):
    """Checks if values are all of desired type.

    Parameters
    ----------
    comparison_type : type
        Type which args are compared against.

    args : iterable
        Arguments user is comparing.

    Returns
    -------
    None
    """
    for a in args:
        if not isinstance(a, comparison_type):
            raise TypeError(F'Passed arg must be type {comparison_type} but type {type(a)} was passed.')


# ----------------------------------------------------------------------------------------------------------------------
