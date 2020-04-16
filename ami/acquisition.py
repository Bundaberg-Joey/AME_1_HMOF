"""
Contains functions used to select the next point for investigation.
Named functions calculate the associated alpha value(s) and the selector function picks uses the alpha values to select points.
"""

import numpy as np
from scipy.stats import norm


def select_datum(untested, pos_mu):
    """Given the alpha values for each member of the dataset and list of untested points,
    select the next point for investigation with the highest posterior mean which has not yet been tested.

    Parameters
    ----------
    untested : list / np.array(), shape(num_not_tested, )
        Integer indices of the data points which have not yet been tested / sampled.

    pos_mu : np.array()
        Array containing the posterior means of each point within the data set.

    Returns
    -------
    ipick : int
        Integer index of the data point to be selected from the original data set.
    """
    max_untested_alpha = np.argmax(pos_mu[untested])
    pick = untested[max_untested_alpha]
    return pick


def random():
    """Select a random point"""
    alpha = np.random.rand(self.n)


def thompson():
    """Selects the point with the highest posterior mean."""
    alpha = self.samples()


def greedy_n():
    """Selects the point with the highest number of instances that it appears in the top N points."""
    y_samples = self.samples(nysamples)
    alpha = np.zeros(self.n)
    for j in range(nysamples):
        alpha[np.argpartition(y_samples[:, j], -N)[-N:]] += 1


def expected_improvement():
    """Selects the point which is expected to provide the greatest improvement to the continually developing model."""
    mu_X_pos, var_X_pos = self.predict()
    sig_X_pos = var_X_pos ** 0.5
    alpha = (mu_X_pos - self.y_max) * norm.cdf(
        np.divide(mu_X_pos - self.y_max, sig_X_pos)) + sig_X_pos * norm.pdf(
        np.divide(mu_X_pos - self.y_max, sig_X_pos))


def greedy_tau():
    """Selects the point with the highest probability of being over the threshold value."""
    if np.mod(self.estimate_tau_counter, self.tau_update) == 0:
        self.estimate_tau()
        self.estimate_tau_counter += 1
    else:
        self.estimate_tau_counter += 1
    mu_X_pos, var_X_pos = self.predict()
    alpha = 1 - norm.cdf(np.divide(self.tau - mu_X_pos, var_X_pos ** 0.5))


def estimate_tau(self, nsamples=10, N=100):
    """
    Used for the `Greedy_tau` sampling algorithm to estimate the value for 'tau'.
    Here 'tau' is the threshold target value for a data point to be considered in the top `N` points.
    This is estimated by calculating the posterior median of threshold to be in the top `N`.

    As this is integrated within an adaptive grid search, the value of `tau` should be recalculated periodically
    in order to shift the value in line with the continually changing posterior distribution.

    Parameters
    ----------
    nsamples : int (default = 10)
        The number of samples to draw from the posterior distribution.

    N : int (default = 100)
        The top n points from which the threshold median value is calculated.

    Returns
    -------
    None :
        updates object attribute `self.tau` (float)
    """
    samples_X_pos = self.samples(nsamples)
    taus = np.zeros(nsamples)
    for i in range(nsamples):
        taus[i] = np.sort(samples_X_pos[:, i])[-N]
    self.tau = np.median(taus)
