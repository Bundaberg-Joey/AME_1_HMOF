"""
Contains functions used to select the next point for investigation.
Named functions calculate the associated alpha values which are maximised when selecting next points.
Currently included are:
* random
* thompson
* expected improvement
* greedy n
* greedy tau
* select_max_alpha
* estimate_tau
"""

import numpy as np
from scipy.stats import norm

from ami import _checks


def random(num_dataset_entries):
    """Generate random alpha values for each entry in the dataset

    Parameters
    ----------
    num_dataset_entries : int
        Number of entries in the dataset being investigated.

    Returns
    -------
    alpha : np.array(), shape(num_dataset_entries, )
        Randomised posterior means, one value per dataset entry
    """
    _checks.pos_int(num_dataset_entries)

    alpha = np.random.randn(num_dataset_entries)
    return alpha


# ----------------------------------------------------------------------------------------------------------------------


def thompson(posterior):
    """Thompson sampling selects the next data point to investigate which has the highest posterior mean.
    As no transformations are required on the `posterior_mean`, the function simply returns the means.
    The function is inlcuded for readability and consistency within the `ami` module.

    Parameters
    ----------
    posterior : np.array(), shape(num_database_entries, 1)
        Vector column of posterior means, achived by sampling posterior through `ami.samples(1)`.

    Returns
    -------
    alpha : np.array(), shape(num_database_entries, 1)
        Posterior means of each entry in the dataset.
    """
    _checks.array_not_empty(posterior)
    _checks.nan_present(posterior)

    alpha = posterior
    return alpha


# ----------------------------------------------------------------------------------------------------------------------


def greedy_n(posterior, n):
    """Greedy N sampling determines the alpha values by the frequency that they appear in the top `n` points.
    The posterior is sampled `n` times so that each data point has `n` associated posterior values.
    Each time that a datapoint has an entry in the top `n`, the alpha values increase by one `increment`.
    The maximum alpha value therefore is the data point with the highest number of appearences in the top `n`.

    Parameters
    ----------
    posterior : np.array(), shape(num_dataset_entries, num_posterior_samples)
        Posterior sampling of the dataset for each entry.
        Each entry (row) has `n` associated values / draws from the posterior.
        Dataset must have been sampled `b` times which can be achieved with `ami.samples(b)`

    n : int
        The number of data points which the greedy `n` algorithm is attempting to optimise for.
        i.e. if `n` = 100, calculates the number of times each data point appears in the top 100.

    Returns
    -------
    alpha : np.array(), shape(num_dataset_entries, )
        Array containing the counts for each datapoint that it appeard in the top `n` data points in the dataset.
        If a value is not present in the top `n` then it will have a default value of 0.
    """
    _checks.array_not_empty(posterior)
    _checks.nan_present(posterior)
    _checks.pos_int(n)

    a, b = posterior.shape
    alpha = np.zeros(a)
    for j in range(b):
        alpha[np.argpartition(posterior[:, j], -n)[-n:]] += 1  # increment count by 1
    return alpha


# ----------------------------------------------------------------------------------------------------------------------


def expected_improvement(mu_pred, var_pred, y_max):
    """EI sampling, selects the next point based on its expectation to improve the most over the current highest found.
    i.e. it selects the point which has the highest probability of having a higher target value than currently known.

    The improvement is found by comparing the model predictions and the highest value currently found.
    The difference between the two arrays is therefore all the possible improvements which could be made based on the
    model's current predictions.

    The expectation is the probability of the scaled model predictions within a standard normal distributon.

    Parameters
    ----------
    mu_pred : np.array(), shape(num_dataset_entries, )
        Predicted means of all entries in the dataset.
        Can be acquired through calling `ami.predict()`.

    var_pred : np.array(), shape(num_dataset_entries, )
        Predicted variance of all entries in the dataset.
        Can be acquired through calling `ami.predict()`.

    y_max : float
        Highest true value which has been investigated by the model.

    Returns
    -------
    alpha : np.array(num_dataset_entries, )
        Array of expected improvement alpha values for each entry in the dataset.
    """
    _checks.array_not_empty(mu_pred, var_pred)
    _checks.nan_present(mu_pred, var_pred)
    _checks.any_float(y_max)

    sig_pred = np.sqrt(var_pred)
    improvement = mu_pred - y_max
    scaled_mu = np.divide(improvement, sig_pred)
    alpha = improvement * norm.cdf(scaled_mu) + sig_pred * norm.pdf(scaled_mu)
    return alpha


# ----------------------------------------------------------------------------------------------------------------------


def greedy_tau(mu_pred, var_pred, tau):
    """Greedy Tau sampling selects the point with the highest probability of being over a given target threshold `tau`.
    The threshold `tau` is calculated from the model posterior and hence updates as the model investigates further.
    Therefore the associated probabilities of data points being over the threshold will shift during investigation.

    The algorithm is considered "greedy" since `tau` is typically the threshold for a data point to be in the top `n`
    data points for the target.
    This differs from `greedy_n` sampling as here we state the threshold rather than the num allowed consituents.
    Accordingly `tau` should be periodically updated as the investigation continues.

    Parameters
    ----------
    mu_pred : np.array(), shape(num_dataset_entries, )
        Predicted means of all entries in the dataset.
        Can be acquired through calling `ami.predict()`.

    var_pred : np.array(), shape(num_dataset_entries, )
        Predicted variance of all entries in the dataset.
        Can be acquired through calling `ami.predict()`.

    tau : float
        The threshold value for a data point to belong to the top `n` target values.


    Returns
    -------
    alpha : np.array(num_dataset_entries, )
        Associated probabilities of each data point to surpass the threshold (`tau`) value.
    """
    _checks.array_not_empty(mu_pred, var_pred)
    _checks.nan_present(mu_pred, var_pred)
    _checks.any_float(tau)

    sig_pred = np.sqrt(var_pred)
    thresh_scaled = np.divide(tau - mu_pred, sig_pred)
    alpha = 1 - norm.cdf(thresh_scaled)
    return alpha

# ----------------------------------------------------------------------------------------------------------------------


def select_max_alpha(untested, alpha):
    """Given the alpha values for each member of the dataset and list of untested points,
    select the next point for investigation with the highest alpha value which has not yet been tested.

    Parameters
    ----------
    untested : list / np.array(), shape(num_not_tested, )
        Integer indices of the data points which have not yet been tested / sampled.

    alpha : np.array(), shape(num_database_entries, )
        Array containing the alpha values of each point within the data set.
        Alpha values are calculated using the acquisution functions in `ami.acquisition`.

    Returns
    -------
    ipick : int
        Integer index of the data point to be selected from the original data set.
    """
    _checks.array_not_empty(untested, alpha)
    _checks.nan_present(untested, alpha)

    max_untested_alpha = np.argmax(alpha[untested])
    pick = untested[max_untested_alpha]
    return pick


# ----------------------------------------------------------------------------------------------------------------------


def estimate_tau(posterior, n):
    """Used to calculate `tau` for `greedy_tau` sampling where `tau` is the threshold value for a data point to be
    in the top `n` data points.
    The threshold values for the top `n` data points for each sampling of the posterior are pooled and the returned
    tau value is the median of those thresholds.

    Note : As estimateing the `tau` value is not an inexpensive calculation, it is recommended to only update `tau`
    perdiodically rather than every model investigation iteration.

    Parameters
    ----------
    posterior : np.array(), shape(num_dataset_entries, num_posterior_samples)
        Posterior sampling of the dataset for each entry.
        Each entry (row) has `n` associated values / draws from the posterior.
        Dataset must have been sampled `n` times which can be achieved with `ami.samples(n)`.

    n : int
        The number of data points which the greedy `tau` algorithm is attempting to optimise for.
        i.e. if `n` = 100, calculates the number of times each data point appears in the top 100.

    Returns
    -------
    tau : float
        The median of the `n`th top data points from the sampled posterior.
    """
    _checks.array_not_empty(posterior)
    _checks.nan_present(posterior)
    _checks.pos_int(n)

    a, b = posterior.shape

    taus = np.zeros(b)
    for i in range(b):
        taus[i] = np.sort(posterior[:, i])[-n]

    tau = np.median(taus)
    return tau
