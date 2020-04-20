"""
Used for checking user assignemnt of values.
Functions ensure correct type and value of values are pased.
Used for setters, etc.
"""


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
