
import pytest
import numpy as np

from ami import alpha


# alpha.random ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_dataset_entries", [1, 10, 100])
def test_random_pass(num_dataset_entries):
    """Returned is a numpy array with floating values of correct size.
    """
    a = alpha.random(num_dataset_entries)
    assert isinstance(a, np.ndarray)
    assert a.dtype == 'float'
    assert a.size == num_dataset_entries


@pytest.mark.xfail(reason="Invalid inputs")
@pytest.mark.parametrize("num_dataset_entries", ['a', [1], 1.0, {}, -1])
def test_random_fail(num_dataset_entries):
    """Internal checks should cause all to fail.
    """
    a = alpha.random(num_dataset_entries)


# alpha.thompson -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("posterior", 
                         [
                             np.random.randn(100),
                             np.array(1),
                             np.random.randn(20, 20)
                         ])
def test_thompson_pass(posterior):
    """Should return what went in, so confirm output is identical to input.
    """
    a = alpha.thompson(posterior)
    assert isinstance(a, np.ndarray)
    assert np.array_equal(a, posterior)


@pytest.mark.xfail(reason="Invalid inputs")
@pytest.mark.parametrize("posterior",
                         [
                             np.array([]),
                             np.array([1, np.nan])
                         ])
def test_thompson_fail(posterior):
    """Internal checks should cause all to fail.
    """
    a = alpha.thompson(posterior)


# alpha.greedy_n -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("posterior, n",
                         [
                             (np.random.randn(100, 20), 5),
                             (np.random.randn(1000, 1), 3),
                             (np.random.randn(1000, 20), 100)
                         ])
def test_greedyn_pass(posterior, n):
    """1d array returned containing increments, must pass 2D array with > 1 rows.
    """
    a = alpha.greedy_n(posterior, n)
    assert isinstance(a, np.ndarray)
    assert a.ndim == 1
    assert a.max() <= n  # no count should be higher than the number of times checked
    assert a.min() >= 0  # values should not be less than 0


@pytest.mark.xfail(reason="Invalid inputs")
@pytest.mark.parametrize("posterior, n",
                         [
                             (np.random.randn(1, 1000), 3),  # incorrect dimentions
                             (np.random.randn(100, 20), -5),  # negative int
                             (np.full((100, 2), (np.random.randn(), np.nan)), 5),  # nan
                             (np.array([]), 3),  # empty
                         ])
def test_greeyn_fail(posterior, n):
    """Internal checks should cause all to fail.
    """
    a = alpha.greedy_n(posterior, n)


# alpha.expected_improvement -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mu_pred, var_pred, y_max",
                         [
                             (np.random.normal(130, 50, 100), np.random.poisson(50, 100), 150.0),
                             (np.random.normal(20, 5, 100), np.random.poisson(10, 100), 10),
                             (10, 5, 3)
                         ])
def test_ei_pass(mu_pred, var_pred, y_max):
    """Checks that input and outputs match, no values less than 0 and no `nan` generated
    Poisson distributions for variance as can't have negative variance.
    """
    a = alpha.expected_improvement(mu_pred, var_pred, y_max)
    a = np.array(a)
    assert a.shape == np.array(mu_pred).shape  # ensure input and output shapes are maintained
    assert a.min() > 0  # shouldn't have negative alpha values
    assert np.isnan(a).sum() == 0  # shouldn't have any nan values in output if none going in


@pytest.mark.xfail(reason="Invalid inputs")
@pytest.mark.parametrize("mu_pred, var_pred, y_max",
                         [
                             ([1, 2, 3], [4, 5, 6], 1.0),  # list not accepted
                             (np.array([1]), np.array([1, 2, 3, 4, 5]), 10),  # different sized arrays
                             (np.array([]), np.array([]), 150),  # empty arrays
                             (np.array([1, 2, np.nan]), np.random.randn(3), 1),  # nan present
                             (np.random.normal(130, 50, 100), np.random.poisson(50, 100), '100'),  # non numeric y_max
                             (np.random.normal(130, 50, 100), np.random.normal(50, 40, 100), '100')  # will generate nan
                         ])
def test_ei_fail(mu_pred, var_pred, y_max):
    """Internal logic should handle most checks, but will return nans when negative variance passed.
    """
    a = alpha.expected_improvement(mu_pred, var_pred, y_max)
    assert np.isnan(a).sum() == 0
