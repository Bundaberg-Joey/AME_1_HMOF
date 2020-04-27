
import pytest
import numpy as np

from ami import _checks


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


# _checks.any_float ---------------------------------------------------------------------------------------------------
def test_anyfloat_pass():
    _checks.any_float(1.0)
    _checks.any_float(-1.0, 0.0, 1.0)


@pytest.mark.xfail(reason='Invalid inputs')
def test_anyfloat_fail():
    _checks.any_float(1)
    _checks.any_float(1.0, -1)
    _checks.pos_int('1')
    _checks.pos_int([1])
    _checks.pos_int({1: 1})


# _checks.any_numeric --------------------------------------------------------------------------------------------------
def test_anynumeric_pass():
    _checks.any_numeric(1)
    _checks.any_numeric(1.0)
    _checks.any_numeric(-1, 0.0, 1, 2.1, -3.1)


@pytest.mark.xfail(reason='Invalid inputs')
def test_anynumeric_fail():
    _checks.any_numeric('1')
    _checks.any_numeric([1], 1)
    _checks.any_numeric({1: 1}, 1)


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
def test_boolean_pass():
    _checks.boolean(True)
    _checks.boolean(False)
    _checks.boolean(True, False, True)


@pytest.mark.xfail(reason='Invalid, inputs')
def test_boolean_fail():
    _checks.boolean(1)
    _checks.boolean(0)
    _checks.boolean(1, 0, 1)
    _checks.boolean(1.0, 0.0)
