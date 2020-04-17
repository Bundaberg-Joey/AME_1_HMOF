"""
Contains utility functions for ami investigations.
Currently included are:
* select max alpha
* estimate tau
"""

import numpy as np


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
    max_untested_alpha = np.argmax(alpha[untested])
    pick = untested[max_untested_alpha]
    return pick


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
    a, b = posterior.shape

    taus = np.zeros(b)
    for i in range(b):
        taus[i] = np.sort(posterior[:, i])[-n]

    tau = np.median(taus)
    return tau
