
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
def test_greedyn_fail(posterior, n):
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
    assert a.min() >= 0  # shouldn't have negative alpha values
    assert np.isnan(a).sum() == 0  # shouldn't have any nan values in output if none going in


@pytest.mark.xfail(reason="Invalid inputs")
@pytest.mark.parametrize("mu_pred, var_pred, y_max",
                         [
                             ([1, 2, 3], [4, 5, 6], 1.0),  # list not accepted
                             (np.array([1]), np.array([1, 2, 3, 4, 5]), 10),  # different sized arrays
                             (np.array([]), np.array([]), 150),  # empty arrays
                             (np.array([1, 2, np.nan]), np.random.randn(3), 1),  # nan present
                             (np.random.normal(130, 50, 100), np.random.poisson(50, 100), '100'),  # non numeric y_max
                         ])
def test_ei_fail(mu_pred, var_pred, y_max):
    """Internal logic should handle most checks.
    """
    a = alpha.expected_improvement(mu_pred, var_pred, y_max)


# alpha.greedy_tau -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mu_pred, var_pred, tau",
                         [
                             (np.random.normal(130, 50, 100), np.random.poisson(50, 100), 150.0),
                             (np.random.normal(20, 5, 100), np.random.poisson(10, 100), 10),
                             (10, 5, 3)
                         ])
def test_greedytau_pass(mu_pred, var_pred, tau):
    """Checks that input and outputs match, no values less than 0 and no `nan` generated
    Poisson distributions for variance as can't have negative variance.
    """
    a = alpha.greedy_tau(mu_pred, var_pred, tau)
    a = np.array(a)
    assert a.shape == np.array(mu_pred).shape  # ensure input and output shapes are maintained
    assert np.isnan(a).sum() == 0  # shouldn't have any nan values in output if none going in
    assert a.min() >= 0  # alpha values should be between 0 and 1
    assert a.max() <= 1


@pytest.mark.xfail(reason="Invalid inputs")
@pytest.mark.parametrize("mu_pred, var_pred, tau",
                         [
                             ([1, 2, 3], [4, 5, 6], 1.0),  # list not accepted
                             (np.array([1]), np.array([1, 2, 3, 4, 5]), 10),  # different sized arrays
                             (np.array([]), np.array([]), 150),  # empty arrays
                             (np.array([1, 2, np.nan]), np.random.randn(3), 1),  # nan present
                             (np.random.normal(130, 50, 100), np.random.poisson(50, 100), '100'),  # non numeric y_max
                         ])
def test_greedytau_fail(mu_pred, var_pred, tau):
    """Internal logic should handle most checks.
    """
    a = alpha.greedy_tau(mu_pred, var_pred, tau)


# alpha.select_max_alpha -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("untested, al, expected",
                         [
                             ([6, 10, 2, 11], np.arange(0, 12, 1), 11),
                             ([0, 10, 2, 11], np.arange(11, -1, -1), 0),
                             ([6, 10, 2, 11], np.arange(11, -1, -1), 2)
                         ])
def test_selectmaxalpha_pass(untested, al, expected):
    """Checks only one integer returned and that it is the expected value.
    """
    pick = alpha.select_max_alpha(untested, al)
    assert isinstance(pick, int)  # only return an integer
    assert pick == expected  # get out what we expected


@pytest.mark.xfail(reason="Invalid inputs")
@pytest.mark.parametrize("untested, al",
                         [
                             ([], np.arange(0, 12, 1)),  # empty array
                             ([0, np.nan, 2, 11], np.arange(11, -1, -1)),  # nan present
                         ])
def test_selectmaxalpha_pass(untested, al):
    """Internal logic should handle most checks.
    """
    pick = alpha.select_max_alpha(untested, al)


# alpha.estimate_tau ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("posterior, n",
                         [
                             (np.random.randn(100, 20), 5),
                             (np.random.randn(1000, 1), 3),
                             (np.random.randn(1000, 20), 100)
                         ])
def test_estimatetau_pass(posterior, n):
    """float returned, must pass 2D array with > 1 rows.
    """
    a = alpha.estimate_tau(posterior, n)
    assert isinstance(a, float)


@pytest.mark.xfail(reason="Invalid inputs")
@pytest.mark.parametrize("posterior, n",
                         [
                             (np.random.randn(1, 1000), 3),  # incorrect dimentions
                             (np.random.randn(100, 20), -5),  # negative int
                             (np.full((100, 2), (np.random.randn(), np.nan)), 5),  # nan
                             (np.array([]), 3),  # empty
                         ])
def test_estimatetau_fail(posterior, n):
    """Internal checks should cause all to fail.
    """
    a = alpha.estimate_tau(posterior, n)


# ----------------------------------------------------------------------------------------------------------------------
