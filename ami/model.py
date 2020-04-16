#!/usr/bin/env python3

"""
Sparse GaussianProcess for use in high dimensional large scale Bayesian Optimization problems.
Fits hyperparameters using dense GPy model.
Special routines for sparse sampling from posterior.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import GPy


class Prospector(object):
    """
    Adaptive grid searching algorithm combining gaussian process regression (RBF kernel) and bayesian optimisation.

    Attributes
    ----------
    X : np.array(), shape (num entries, num features)
        Feature matrix containing numerical values.

    n : int
        Number of rows in the feature matrix `X`.

    d : int
        Number of columns in the feature matrix `X`.

    update_counter : int (default = 0)
        Counter to track the number of model iterations which have occured.

    updates_per_big_fit : int (default = 0)
        The number of model iterations between sampling and fully fitting model hyperparameters.
        When the sample is made on a non `update_per_big_fit` iteration then model just fits to data.


    `None` initialised attributes
    -----------------------------
    y_max : float
        The maximum obtained target value.

    GP : GPy.models.gp_regression.GPRegression
        Gaussian Process Regressor (GPR) model from `GPy` which has been fit to the training data.

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
    fit(Y, STATUS, ntop=100, nrecent=100, nmax=400, ntopmu=100, ntopvar=100, nkmeans=300, nkeamnsdata=5000,
            lam=1e-6)

    predict(nsamples=1)

    samples(nsamples=10, N=100)

    estimate_tau(nsamples=10, N=100)

    predict(STATUS, N=100, nysamples=100)

    Notes
    -----
    Used to identify the top ranking members of a dataset for a passed target by an adaptive grid searching approach.
    The model predictions are based on gaussian process regression, allowing the predicted mean and variance values to
    drive the selection of the next dataset member to investigate.
    Numerous selection algorithms are implemented and are selected by passing the keyword to the init function: []
    Nystrom inducing points inducing points that are also actual trainingdata points) are used to speed up the fitting
    of the large covariance matrices used by sparse inference.
    """

    def __init__(self, X):
        """
        Parameters
        ----------
        X : np.array(), shape (num entries, num features)
            Feature matrix containing numerical values.
        """
        self.X = X
        self.n, self.d = X.shape
        self.update_counter = 0
        self.updates_per_big_fit = 10
        self.y_max = None
        self.GP = None
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

    def fit(self, Y, STATUS, ntop=100, nrecent=100, nmax=400, ntopmu=100, ntopvar=100, nkmeans=300, nkeamnsdata=5000,
            lam=1e-6):
        """
        Fits model hyperparameter and inducing points using a GPy dense model to determine hyperparameters.

        Each time `fit` is run, the number of points assessed is calculated. If it is greater than `nmax` then to
        conserve computational power only a subsample of points are considered.
        The subsample will contain the `nrecent', `ntopmu` points and (`nmax`-('nrecent'+'ntopmu')) random points not
        present in the `nrecent` or `ntopmu` points.

        The `GPy` library is used to fit the base model parameters {`self.mu`, `self.a`, `self.l`, `self.b`} which
        minimises the NLL on the training data (points which have been tested and so have true values available). The
        `GPy` library is yet unable to perform high(ish) dim sparse inference so the base model parameters are extracted
        and used as internal attributes.

        In addition to the nystrom inducing points, `KMeansCLustering` with `nkmeans` clusters is used to get some
        nontested inducing points spread throughout the domain.
        As clustering can take a significant amount of time, a uniformly random subsample of `nkmeansdata` points from
        the untested points is used here.
        To ensure the model's length scales are factored into the clustering (euclidean distance) the coordinates are
        scaled using the length scales `self.l` to ensure an appropriate distance measure is used.

        NOTE : Contact Dr Hook by email `james.l.hook@gmail.com` if serious issues develop during the fitting process,

        Parameters
        ----------
        Y : np.array(), shape(num_entries,)
            True values of target points selected to be assessed.

        STATUS : np.array(), shape(num_entries,1)
            Vector which tracks which points have been assessed or not and to what degree.

        ntop : int (default = 100)
            The top n true points to consider when performing the fit on subset.

        nrecent : int (default = 100)
            The number of recent samples to consider when performing fit on subset.

        nmax : int (default = 400)
            The maximum number of random samples to be taken by the model .

        ntopmu : int (default = 100)
            The number of untested points which the model predicts to be the highest ranked.
            Used to generate inducing points.

        ntopvar : int (default = 100)
            The number of untested points which the model has the highest uncertainty values for.
            Used to generate inducing points.

        nkmeans : int (default = 300)
            Number of clusters to consider when using KMeansClustering when determining inducing points.

        nkeamnsdata : int (default = 5000)
            Number of points fed into the KMeansCLustering algorithm for determining inducing points.

        lam : float (default = 1e-6)
            Controls the jitter in samples when determining the covariance matrix `SIG_MM`

        Returns
        -------
        None :
            Updates object attributes {`self.SIG_XM`, `self.SIG_MM`, `self.SIG_MM_pos`, `self.SIG_M_pos`}
        """
        X = self.X
        untested = [i for i in range(self.n) if STATUS[i] == 0]
        tested = [i for i in range(self.n) if STATUS[i] == 2]
        ytested = Y[tested].reshape(-1)
        self.y_max = np.max(ytested)

        if np.mod(self.update_counter, self.updates_per_big_fit) == 0:
            print('fitting hyperparameters')
            ntested = len(tested)

            if ntested > nmax:
                top = list(np.argsort(ytested)[-ntop:])
                recent = list(range(ntested - nrecent, ntested))
                topandrecent = list(set(top + recent))
                rand = list(
                    np.random.choice([i for i in range(ntested) if i not in topandrecent], nmax - len(topandrecent),
                                     False))
                testedtrain = topandrecent + rand
                ytrain = ytested[testedtrain]
                train = [tested[i] for i in testedtrain]
            else:
                train = tested
                ytrain = ytested

            mfy = GPy.mappings.Constant(input_dim=self.d, output_dim=1)
            ky = GPy.kern.RBF(self.d, ARD=True, lengthscale=np.ones(self.d))
            self.GP = GPy.models.GPRegression(X[train], ytrain.reshape(-1, 1), kernel=ky, mean_function=mfy)
            self.GP.optimize('bfgs')
            self.mu = self.GP.flattened_parameters[0]
            self.a = self.GP.flattened_parameters[1]
            self.l = self.GP.flattened_parameters[2]
            self.b = self.GP.flattened_parameters[3]

            print('selecting inducing points')
            self.py = self.GP.predict(X)
            topmu = [untested[i] for i in np.argsort(self.py[0][untested].reshape(-1))[-ntopmu:]]
            topvar = [untested[i] for i in np.argsort(self.py[1][untested].reshape(-1))[-ntopvar:]]
            nystrom = topmu + topvar + train
            kms = KMeans(n_clusters=nkmeans, max_iter=5).fit(
                np.divide(X[list(np.random.choice(untested, nkeamnsdata))], self.l))
            self.M = np.vstack((X[nystrom], np.multiply(kms.cluster_centers_, self.l)))

            print('fitting sparse model')
            DXM = euclidean_distances(np.divide(X, self.l), np.divide(self.M, self.l), squared=True)
            self.SIG_XM = self.a * np.exp(-DXM / 2)
            DMM = euclidean_distances(np.divide(self.M, self.l), np.divide(self.M, self.l), squared=True)
            self.SIG_MM = self.a * np.exp(-DMM / 2) + np.identity(self.M.shape[0]) * lam * self.a
            self.B = self.a + self.b - np.sum(np.multiply(np.linalg.solve(self.SIG_MM, self.SIG_XM.T), self.SIG_XM.T),
                                              0)
            K = np.matmul(self.SIG_XM[tested].T, np.divide(self.SIG_XM[tested], self.B[tested].reshape(-1, 1)))
            self.SIG_MM_pos = self.SIG_MM - K + np.matmul(K, np.linalg.solve(K + self.SIG_MM, K))
            J = np.matmul(self.SIG_XM[tested].T, np.divide(ytested - self.mu, self.B[tested]))
            self.mu_M_pos = self.mu + J - np.matmul(K, np.linalg.solve(K + self.SIG_MM, J))

        else:
            K = np.matmul(self.SIG_XM[tested].T, np.divide(self.SIG_XM[tested], self.B[tested].reshape(-1, 1)))
            self.SIG_MM_pos = self.SIG_MM - K + np.matmul(K, np.linalg.solve(K + self.SIG_MM, K))
            J = np.matmul(self.SIG_XM[tested].T, np.divide(ytested - self.mu, self.B[tested]))
            self.mu_M_pos = self.mu + J - np.matmul(K, np.linalg.solve(K + self.SIG_MM, J))

        self.update_counter += 1

    def predict(self):
        """
        Calculates the predicted mean and predicted variance of the full dataset.
        This utilises the covarince matrices of the prior `self.SIG_MM` and posterior `self.SIG_XM` distributions.
        For calculting the predicted mean of the posterior, a mean shifted process is used to facilite more reliable
        predictions of datapoints with process means != 0. Otherwise a mean of 0 would have to be assumed or the process
        mean would be based on a sample of the original dataset which would depend on the sampling conducted and likely
        not be representative.

        Returns
        -------
        mu_X_pos : np.array(), shape(num_entries, )
            The predicted means of each point in the dataset.

        var_X_pos : np.array(), shape(num_entries, )
            The predicted variance of each point in the dataset.
        """
        mu_X_pos = self.mu + np.matmul(self.SIG_XM, np.linalg.solve(self.SIG_MM, self.mu_M_pos - self.mu))
        var_X_pos = np.sum(np.multiply(
            np.matmul(np.linalg.solve(self.SIG_MM, np.linalg.solve(self.SIG_MM, self.SIG_MM_pos).T), self.SIG_XM.T),
            self.SIG_XM.T), 0)
        return mu_X_pos, var_X_pos

    def samples(self, nsamples=1):
        """
        Conducts a sparse sampling of the dataset by sampling on the calculated inducing points and then uses the
        conditional mean given sample values on the full dataset.

        Parameters
        ----------
        nsamples : int (default = 1)
            The number of samples to draw from the posterior distribution.

        Returns
        -------
        samples_X_pos : np.array()
            Matrix whose columns are independent samples of the posterior over the full dataset
        """
        samples_M_pos = np.random.multivariate_normal(self.mu_M_pos, self.SIG_MM_pos, nsamples).T
        samples_X_pos = self.mu + np.matmul(self.SIG_XM, np.linalg.solve(self.SIG_MM, samples_M_pos - self.mu))
        return samples_X_pos

