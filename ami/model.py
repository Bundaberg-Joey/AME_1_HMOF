#!/usr/bin/env python3

"""
Sparse GaussianProcess for use in high dimensional large scale Bayesian Optimization problems.
Fits hyperparameters using dense GPy model.
Special routines for sparse sampling from posterior.
"""

# TODO : rename the attribues of this thing... yeesh

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from GPy.mappings import Constant
from GPy.kern import RBF
from GPy.models import GPRegression

from ami import _checks


class Prospector(object):
    """Adaptive grid searching algorithm combining gaussian process regression (RBF kernel) and bayesian optimisation.


    Attributes
    ----------
    X : np.array(), shape (num entries, num features)
        Feature matrix containing numerical values.

    cluster_func : clustering algorithm (default = KMeans(n_clusters=300, max_iter=5))
        Clustering algorithm used to select inducing points.
        Must have the below functionality:
            * `fit(np.array(num_entries, num_features)` --> determine clusters for data and store internally
            * `cluster_centers_` --> np.array(), shape(num_entries, num_features)

    n : int
        Number of rows in the feature matrix `X`.

    d : int
        Number of columns in the feature matrix `X`.

    ntopmu : int (default = 100)
        The number of untested points which the model predicts to be the highest ranked.
        Used to generate inducing points when fitting model.

    ntopvar : int (default = 100)
        The number of untested points which the model has the highest uncertainty values for.
        Used to generate inducing points when fitting model.

    n_points_to_cluster : int (default = 5000)
        Number of points fed into the KMeansCLustering algorithm for determining inducing points for fit.

    lam : float (default = 1e-6)
        Controls the jitter in samples when determining the covariance matrix `SIG_MM` when fitting.


    `None` initialised attributes
    -----------------------------
    mu : np.array()
        Prior mean as predicted by the the `GPy` GPR model.

    a : np.array()
        RBF kernel variance as predicted by the the `GPy` GPR model.

    l : np.array()
        Feature length scales as determined by the the `GPy` GPR model.

    b : np.array()
        Prior variance as predicted by the the `GPy` GPR model.

    py : np.array(), shape(num entries, )
        Target values as predicted by the 'GPy` GPR model.

    M : np.array()
        Matrix of inducing points used for fitting the model.

    SIG_XM : np.array()
        The prior covarience matrix between data and inducing points.

    SIG_MM : np.array()
        The prior covarience matrix at inducing points.

    SIG_MM_pos : np.array()
        The posterior covarience matrix at inducing points.

    mu_M_pos : np.array()
        The posterior mean at inducing points.

    B : np.array()
        USed for fitting of model on sparse induction points with high dimensionality.


    Methods
    -------
    fit_posterior(ytested, tested) --> Update posterior distribution with new data points.
    update_model_parameters(untested, train, ytrain) --> Update the hyperparams / inducing points / covariance matrices
    predict(nsamples=1) --> predict the mean and variance for each data point.
    sample_posterior(self, n_repeats=1) --> draw samples from the posterior with n repetitions

    property getter / setters:
        * ntopmu
        * ntopvar
        * n_points_to_cluster


    Notes
    -----
    Used to identify the top ranking members of a dataset for a passed target by an adaptive grid searching approach.
    The model predictions are based on gaussian process regression, allowing the predicted mean and variance values to
    drive the selection of the next dataset member to investigate.
    Nystrom inducing points inducing points that are also actual trainingdata points) are used to speed up the fitting
    of the large covariance matrices used by sparse inference.
    """

    def __init__(self, X, cluster_func=KMeans(n_clusters=300, max_iter=5)):
        """
        Parameters
        ----------
        X : np.array(), shape (num entries, num features)
            Feature matrix containing numerical values.

        cluster_func : clustering algorithm (default = KMeans(n_clusters=300, max_iter=5))
            Clustering algorithm used to select inducing points.
            Must have the below functionality:
                * `fit(np.array(num_entries, num_features)` --> determine clusters for data and store internally
                * `cluster_centers_` --> np.array(), shape(num_entries, num_features)

        """
        _checks.array_not_empty(X)
        _checks.nan_present(X)

        self.X = X
        self.cluster_func = cluster_func
        self.n, self.d = X.shape
        self.ntopmu = 100
        self.ntopvar = 100
        self.n_points_to_cluster = 5000
        self.lam = 1e-6
        self.mu = None
        self.a = None
        self.l = None
        self.b = None
        self.py = None
        self.M = None
        self.SIG_XM = None
        self.SIG_MM = None
        self.SIG_MM_pos = None
        self.mu_M_pos = None
        self.B = None

    def _determine_prior_covariance_matrices(self):
        """Updates the prior covariance matrices of both the feature set and inducing points.
        Rescales feature and inducing point matrices using the model feature length scales as length scale
        will impact feature distance / significance.
        The euclidean distances are then treated with a gaussian kernel and predicted kernel variance to inform
        the relevant matrices.

        Returns
        -------
        None :
            Updates attributes {`self.SIG_XM`, `self.SIG_MM`, 'self.B'}
        """
        scaled_x = np.divide(self.X, self.l)
        scaled_m = np.divide(self.M, self.l)
        distance_xm = euclidean_distances(scaled_x, scaled_m, squared=True)
        distance_mm = euclidean_distances(scaled_m, scaled_m, squared=True)
        self.SIG_XM = self.a * np.exp(-distance_xm / 2)
        self.SIG_MM = self.a * np.exp(-distance_mm / 2) + np.identity(self.M.shape[0]) * self.lam * self.a
        self.B = self.a + self.b - np.sum(np.multiply(np.linalg.solve(self.SIG_MM, self.SIG_XM.T), self.SIG_XM.T), 0)

    def _select_inducing_points(self, untested, train):
        """Determines optimal location of inducing points for fitting the sparse gaussian model.

        Uses `cluster_func` to get nontested inducing points spread through the domain.
        Random uniform sample of `n_points_to_cluster` points from the untested points are taken due to cluster expense.
        To ensure the model's length scales are factored into the clustering (euclidean distance) the coordinates are
        scaled using the length scales `self.l` to ensure an appropriate distance measure is used.

        Parameters
        ----------
        untested : list shape(num_points_untested, )
            Indices of data points in the feature matrix `X` which have not been tested.

        train : list shape(num_points_tested, )
            Indices of data points in the feature matrix `X` which have been tested, to balance inducing points.

        Returns
        -------
        None :
            Updates attribue `self.M`
        """
        topmu = [untested[i] for i in np.argsort(self.py[0][untested].reshape(-1))[-self.ntopmu:]]
        topvar = [untested[i] for i in np.argsort(self.py[1][untested].reshape(-1))[-self.ntopvar:]]
        nystrom = np.concatenate((topmu, topvar, train))

        rand_X = self.X[np.random.choice(untested, self.n_points_to_cluster)]
        rand_scaled_X = np.divide(rand_X, self.l)
        self.cluster_func.fit(rand_scaled_X)

        self.M = np.vstack((self.X[nystrom], np.multiply(self.cluster_func.cluster_centers_, self.l)))

    def _update_hyperparameters(self, train, ytrain):
        """Fits hyperparameters of the gaussian process regressor using external model.

        The `GPy` library is used to fit the base model parameters {`self.mu`, `self.a`, `self.l`, `self.b`} which
        minimises the NLL on the training data (points which have been tested and so have true values available).
        The `GPy` library is yet unable to perform high(ish) dim sparse inference so the base model parameters are
        extracted and used as internal attributes.

        Parameters
        ----------
        train : list shape(num_points_tested, )
            Indices of data points in the feature matrix `X` which have been tested, to balance inducing points.

        ytrain : list shape(num_tested_points, )
            List of target values which have been determined empirically by testing.

        Returns
        -------
        None:
            Updates attributes {`self.mu`, `self.a`, `self.l` `self.b`, `self.py`}
        """
        mfy = Constant(input_dim=self.d, output_dim=1)
        ky = RBF(self.d, ARD=True, lengthscale=np.ones(self.d))
        gpr = GPRegression(self.X[train], ytrain.reshape(-1, 1), kernel=ky, mean_function=mfy)
        gpr.optimize('bfgs')
        self.mu, self.a, self.l, self.b = gpr.flattened_parameters[:4]
        self.py = gpr.predict(self.X)

    def update_model_parameters(self, untested, train, ytrain):
        """Updates the parameters of the Gaussian Process Regressor, inducing points and associated covariance matrices.

        Notes
        -----
        Contact Dr Hook by email `james.l.hook@gmail.com` if serious issues develop during the fitting process.

        Parameters
        ----------
        untested : list / np.array(), shape(num_points_tested, )
            Indices of data points in `X` which have not yet been tested.

        train : list / np.array(), shape(1D)
            Indices of data points n `X` used to determine GP hyperparameters and inducing point locations.
            These are not determined in the same manner as `untested` and so the size of the passed
            array does not always equal `num_points_tested`.

        ytrain : list / np.array(), shape(num_training_points, )
            Emperical values of the data points present in `train`
            The same caveats apply to `ytrain` as `train`

        Returns
        -------
        None :
            Updates numerous attribues concerned with model parameters, covariance matrices and inducing points.
        """
        untested, train, ytrain = np.asarray(untested), np.asarray(train), np.asarray(ytrain)
        _checks.array_not_empty(untested, train, ytrain)
        _checks.nan_present(untested, train, ytrain)

        self._update_hyperparameters(train, ytrain)
        self._select_inducing_points(untested, train)
        self._determine_prior_covariance_matrices()

    def fit_posterior(self, ytested, tested):
        """Incorporates the tested data points in to the Gaussian Process Regressors posterior distribution.
        Posterior mean and covariance matrix at the inducing points are updated.

        Notes
        -----
        Explicitly different from `update_model_parameters` as internal hyperparameters not updated during `fit`.
        Only the posterior distribution is updated here to reduce expense of always updating hyperparameters and inducing points

        Parameters
        ----------
        ytested : np.array(), shape(num_points_tested, )
            Values of target points which have been determined empirically.

        tested : list / np.array(), shape(num_points_tested, )
            List containing indices of data points in `X` which have been tested.

        Returns
        -------
        None :
            Updates attributes {`self.SIG_MM_pos`, `self.SIG_M_pos`}
        """
        _checks.array_not_empty(ytested, tested)
        _checks.nan_present(ytested, tested)

        K = np.matmul(self.SIG_XM[tested].T, np.divide(self.SIG_XM[tested], self.B[tested].reshape(-1, 1)))
        self.SIG_MM_pos = self.SIG_MM - K + np.matmul(K, np.linalg.solve(K + self.SIG_MM, K))
        J = np.matmul(self.SIG_XM[tested].T, np.divide(ytested - self.mu, self.B[tested]))
        self.mu_M_pos = self.mu + J - np.matmul(K, np.linalg.solve(K + self.SIG_MM, J))

    def predict(self, return_variance=True):
        """Calculates the predicted mean and predicted variance of the full dataset.
        This utilises the covarince matrices of the prior `self.SIG_MM` and posterior `self.SIG_XM` distributions.
        For calculting the predicted mean of the posterior, a mean shifted process is used to facilite more reliable
        predictions of datapoints with process means != 0. Otherwise a mean of 0 would have to be assumed or the process
        mean would be based on a sample of the original dataset which would depend on the sampling conducted and likely
        not be representative.

        Parameters
        ----------
        return_variance : bool (default = True)
            If True then variance will calculated and returned with mean.
            Else only mean will be calculated and returned.

        Returns
        -------
        mu_X_pos : np.array(), shape(num_entries, )
            The predicted means of each point in the dataset.

        var_X_pos : np.array(), shape(num_entries, ) (conditionally returned)
            The predicted variance of each point in the dataset.
        """
        _checks.are_type(bool, return_variance)

        mu_X_pos = self.mu + np.matmul(self.SIG_XM, np.linalg.solve(self.SIG_MM, self.mu_M_pos - self.mu))

        if return_variance:
            var_X_pos = np.sum(np.multiply(
                np.matmul(np.linalg.solve(self.SIG_MM, np.linalg.solve(self.SIG_MM, self.SIG_MM_pos).T), self.SIG_XM.T),
                self.SIG_XM.T), 0)
            return mu_X_pos, var_X_pos

        elif not return_variance:
            return mu_X_pos

    def sample_posterior(self, n_repeats=1):
        """Conducts a sparse sampling of the dataset by sampling on the calculated inducing points and then uses the
        conditional mean given sample values on the full dataset.

        Parameters
        ----------
        n_repeats : int (default = 1)
            The number of times the full dataset is to be sampled.

        Returns
        -------
        samples_X_pos : np.array(), shape(num_dataset_entries, n_repeats)
            Matrix whose columns are independent samples of the posterior over the full dataset
        """
        _checks.pos_int(n_repeats)

        samples_M_pos = np.random.multivariate_normal(self.mu_M_pos, self.SIG_MM_pos, n_repeats).T
        samples_X_pos = self.mu + np.matmul(self.SIG_XM, np.linalg.solve(self.SIG_MM, samples_M_pos - self.mu))
        return samples_X_pos

    @property
    def ntopmu(self):
        return self._ntopmu

    @ntopmu.setter
    def ntopmu(self, value):
        """Set value to positive integer. Raise error if incorrect argument type.
        """
        _checks.pos_int(value)
        self._ntopmu = value

    @property
    def ntopvar(self):
        return self._ntopvar

    @ntopvar.setter
    def ntopvar(self, value):
        """Set value to positive integer. Raise error if incorrect argument type.
        """
        _checks.pos_int(value)
        self._ntopvar = value

    @property
    def n_points_to_cluster(self):
        return self._n_points_to_cluster

    @n_points_to_cluster.setter
    def n_points_to_cluster(self, value):
        """Set value to positive integer. Raise error if incorrect argument type.
        """
        _checks.pos_int(value)
        self._n_points_to_cluster = value

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, value):
        """Set value to float. Raise error if incorrect argument type.
        """
        _checks.are_type(float, value)
        self._lam = value


# ----------------------------------------------------------------------------------------------------------------------


class TrainingFilter(object):
    """Determines ideal training points to consider to facilitate sparse fitting of the Prospector.

    Attributes
    ----------
    nmax : int
        The highest number of data points to be allowed in the training set.

    ntop : int
        The number of`top` data points to be included in the training set.

    nrecent : int
        The `n` most recently sampled data points to include in training data.


    Methods
    -------
    select_training_points(self, tested_indices, test_results) --> determines training data.

    Notes
    -----
    As the prospector works by considering nystrom inducing points rather than a full `n` by `n`
    covariance matrix, it is ideal to keep the size of the training data small.
    Here, the training set is contained to an upper limit of `nmax`, which is either the data
    acquired up till now or a more frugal selection.
    The frugal selection combines the `ntop` highest value data points, the `nrecent` training
    points, with the remainder being random data points which have already been tested.
    """

    def __init__(self, nmax=400, ntop=100, nrecent=100):
        """Confirms input variables and raises error if incorrect type or sign.

        Parameters
        ----------
        nmax : int
            The highest number of data points to be allowed in the training set.

        ntop : int
            The number of`top` data points to be included in the training set.

        nrecent : int
            The `n` most recently sampled data points to include in training data.
        """
        _checks.pos_int(nmax, ntop, nrecent)

        self.nmax = nmax
        self.ntop = ntop
        self.nrecent = nrecent

    def select_training_points(self, tested_indices, test_results):
        """Selects training points to use, given imposed constraints on initialisation.
        If the number of tested points is less than the user maximum, then just returns passed arrays to user.

        Parameters
        ----------
        tested_indices : list / array, shape(num_tested_points, )
            Indices of data points in the feature matrix which have been tested.

        test_results : list / array, shape(num_tested_points, )
            Target values of data points which have been tested.

        Returns
        -------
        (train_indices, y_train) : tuple of np.array(), shape( 1<= `nmax`, )
            Indices of data points to use and their emperical target values.
            The size of the arrays are dependant on the conditional size.
        """
        _checks.array_not_empty(tested_indices, test_results)
        _checks.nan_present(tested_indices, test_results)

        tested_indices = np.asarray(tested_indices)
        y_tested = np.asarray(test_results)
        n_tested = len(y_tested)

        if n_tested > self.nmax:
            indices = np.arange(0, n_tested, 1)

            top_ind = np.argsort(y_tested)[-self.ntop:]
            recent_ind = indices[n_tested - self.nrecent:]
            top_recent_ind = np.unique(np.concatenate((top_ind, recent_ind)), axis=0)

            not_toprecent = indices[np.isin(indices, top_recent_ind, invert=True)]
            rand = np.random.choice(not_toprecent, self.nmax - len(top_recent_ind), replace=False)

            testedtrain = np.concatenate((top_recent_ind, rand))
            train_indices, y_train = tested_indices[testedtrain], y_tested[testedtrain]

        else:
            train_indices, y_train = tested_indices, test_results

        return train_indices, y_train
