"""
# todo : add docstring
"""

# todo : add imports

# todo : add functions here for sampling
# todo : estimate tau goes in here also

def pick_next(self, STATUS, N=100, nysamples=100):  # break this down into smaller functions
    """
    Selects the next point to be tested to determine the true value of.
    Several sampling algorithms are implemented below and are selected based on the `acquiisition_function`
    attribute defined at initialisation of the model. If no valid function was selected by the user then the next
    point is randomly selected from all possible points in the dataset.

    Each algorithm relies on different model attributes and selects based on differing criteria:

    `Thompson` : The point with the highest posterior mean.
    `Greedy_N` : The point with the highest number of instances that it appears in the top N points.
    `Greedy_tau` : The point with the highest probability of being over the threshold value.
    `EI` : The point which is expected to provide the greatest improvement to the continually developing model.

    Parameters
    ----------
    STATUS : np.array(), shape(num_entries,1)
        Vector which tracks which points have been assessed or not and to what degree.

    N : int (default = 100)
        The number of materials which the `Greedy_N` algorithm is attempting to optimise for.

    nysamples : int (default = 100)
        The number of samples to sample from posterior for the `Greedy_N` optimisation.

    Returns
    -------
    ipick : int
        The index value of the non tested point in the feature matrix `X`.
    """
    untested = [i for i in range(self.n) if STATUS[i] == 0]

    if self.acquisition_function == 'Thompson':
        alpha = self.samples()

    elif self.acquisition_function == 'Greedy_N':
        y_samples = self.samples(nysamples)
        alpha = np.zeros(self.n)
        for j in range(nysamples):
            alpha[np.argpartition(y_samples[:, j], -N)[-N:]] += 1

    elif self.acquisition_function == 'Greedy_tau':
        if np.mod(self.estimate_tau_counter, self.tau_update) == 0:
            self.estimate_tau()
            self.estimate_tau_counter += 1
        else:
            self.estimate_tau_counter += 1
        mu_X_pos, var_X_pos = self.predict()
        alpha = 1 - norm.cdf(np.divide(self.tau - mu_X_pos, var_X_pos ** 0.5))

    elif self.acquisition_function == 'EI':
        mu_X_pos, var_X_pos = self.predict()
        sig_X_pos = var_X_pos ** 0.5
        alpha = (mu_X_pos - self.y_max) * norm.cdf(
            np.divide(mu_X_pos - self.y_max, sig_X_pos)) + sig_X_pos * norm.pdf(
            np.divide(mu_X_pos - self.y_max, sig_X_pos))

    else:
        alpha = np.random.rand(self.n)

    ipick = untested[np.argmax(alpha[untested])]
    return ipick



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
