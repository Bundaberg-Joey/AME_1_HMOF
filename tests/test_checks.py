
import pytest
import numpy as np

from ami import _checks


# _checks.are_type -----------------------------------------------------------------------------------------------------
def test_aretype_pass():
    _checks.are_type(int, 1)  # single type,  single args
    _checks.are_type(float, 1.0)
    _checks.are_type(str, '1')
    _checks.are_type(list, [1, 2, 3])
    _checks.are_type(dict, {1: '1', 2: '2'})

    _checks.are_type(int, 1, 2, 3)  # single type, multiple args
    _checks.are_type(list, [1, 2, 3], [4, 5, 6])

    _checks.are_type((int, float, list), 1.0)  # multiple types, single arg

    _checks.are_type((str, list), [1, 2, 3], 'hello', 'world')  # multiple types and multiple args


@pytest.mark.xfail(reason='Invalid input')
def test_aretype_fail():
    _checks.are_type(int, 1.0)  # single type, single arg
    _checks.are_type(int, '1', 1.0, [1])  # single type, multiple args
    _checks.are_type((str, int), 3.2, ['Hello', 1])  # multipe types, multiple args


# _checks.pos_int ------------------------------------------------------------------------------------------------------
def test_posint_pass():
    _checks.pos_int(1)
    _checks.pos_int(1, 2, 3, 10000)


@pytest.mark.xfail(reason='Invalid inputs')
def test_posint_fail():
    _checks.pos_int(-1)
    _checks.pos_int(0)
    _checks.pos_int(1, -1.2)
    _checks.pos_int('1')
    _checks.pos_int([1])
    _checks.pos_int({1: 1})


# _checks.any_numeric --------------------------------------------------------------------------------------------------
def test_nanpresent_pass():
    _checks.nan_present(1, 2, 3)
    _checks.nan_present([1, 2], [3])
    _checks.nan_present(np.random.randn(20, 10), np.random.randn(1, 10))


@pytest.mark.xfail(reason='Invalid inputs')
def test_nanpresent_fail():
    _checks.nan_present(np.nan)
    _checks.nan_present([1, 2, 3], [np.nan, 5])
    _checks.nan_present({1: 1})


# _checks.array_not_empty ----------------------------------------------------------------------------------------------
def test_arraynotempty_pass():
    _checks.array_not_empty([1])
    _checks.array_not_empty([1, 2], [3])
    _checks.array_not_empty(np.random.randn(20, 10), np.random.randn(1, 10))
    _checks.array_not_empty(1, 'a', np.nan, {1: 1}, [[1, 2], []])  # known limitations.


@pytest.mark.xfail(reason='Invalid inputs')
def test_arraynotempty_fail():
    _checks.array_not_empty(np.array([]), [])
    _checks.array_not_empty([[], [], []])


# _checks.same_chape ---------------------------------------------------------------------------------------------------
def test_sameshape_pass():
    _checks.same_shape((1, 2.2))
    _checks.same_shape((np.random.randn(10, 6), np.random.randn(10, 6)))
    _checks.same_shape(([[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]))
    _checks.same_shape(({1: 1}, {2: 2, 3: 3, 4: 4}))


@pytest.mark.xfail(reason='Invalid, inputs')
def test_sameshape_fail():
    _checks.same_shape((np.random.randn(10, 3), np.random.randn(3, 10)))
    _checks.same_shape((1, [1]))


# ----------------------------------------------------------------------------------------------------------------------
