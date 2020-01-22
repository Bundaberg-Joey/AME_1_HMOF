#!/usr/bin/env python3

"""
Sparse GaussianProcess for use in high dimensional large scale Bayesian Optimization problems.
Fits hyperparameters using dense GPy model.
Special routines for sparse sampling from posterior.
Thompson and Greedy_N acquisition functions.

"""

__author__ = 'James Hook'
__version__ = '2.0.1'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import GPy


class prospector(object):

    def __init__(self, X):
        """ Initializes by storing all feature values """
        self.X = X
        self.n, self.d = X.shape
        self.update_counter = 0
        self.updates_per_big_fit = 10

    def fit(self, Y, STATUS, ntop=100, nrecent=100, nmax=400, ntopmu=100, ntopvar=100, nkmeans=300, nkeamnsdata=5000,
            lam=1e-6):
        """
        Fits hyperparameters and inducing points.
        Fit a GPy dense model to get hyperparameters.
        Take subsample for tested data for fitting.

        :param Y: np.array(), experimentally determined values
        :param STATUS: np.array(), keeps track of which materials have been assessed / what experiments conducted
        :param ntop: int, top n samples
        :param nrecent: int, most recent samples
        :param nmax: int, max number of random samples to be taken
        :param ntopmu: int, most promising untested points
        :param ntopvar: int, most uncertain untested points
        :param nkmeans: int, cluster centers from untested data
        :param nkeamnsdata: int, ?
        :param lam: float, controls jitter in g samples
        """
        X = self.X
        untested = [i for i in range(self.n) if STATUS[i] == 0]
        tested = [i for i in range(self.n) if STATUS[i] == 2]
        ytested = Y[tested].reshape(-1)
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

            mfy = GPy.mappings.Constant(input_dim=self.d, output_dim=1)  # fit dense GPy model to this data
            ky = GPy.kern.RBF(self.d, ARD=True, lengthscale=np.ones(self.d))
            self.GP = GPy.models.GPRegression(X[train], ytrain.reshape(-1, 1), kernel=ky, mean_function=mfy)
            self.GP.optimize('bfgs')
            self.mu = self.GP.flattened_parameters[0]
            self.a = self.GP.flattened_parameters[1]
            self.l = self.GP.flattened_parameters[2]
            self.b = self.GP.flattened_parameters[3]
            # now pick inducing points for sparse model, use all the train points as above

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
        Get a prediction on full dataset

        :return: mu_X_pos, var_X_pos:
        """
        mu_X_pos = self.mu + np.matmul(self.SIG_XM, np.linalg.solve(self.SIG_MM, self.mu_M_pos - self.mu))
        var_X_pos = np.sum(np.multiply(np.linalg.solve(self.SIG_MM_pos, self.SIG_XM.T), self.SIG_XM.T), 0)
        return mu_X_pos, var_X_pos

    def samples(self, nsamples=1):
        """
        Samples posterior on full dataset

        :param nsamples: int, Number of samples to draw from the posterior distribution

        :return: samples_X_pos: ?
        """
        samples_M_pos = np.random.multivariate_normal(self.mu_M_pos, self.SIG_MM_pos, nsamples).T
        samples_X_pos = self.mu + np.matmul(self.SIG_XM, np.linalg.solve(self.SIG_MM, samples_M_pos - self.mu))
        return samples_X_pos

    def pick_next(self, STATUS, acquisition_function='Thompson', N=100, nysamples=100):
        """

        Picks next material to sample

        :param STATUS: np.array(), keeps track of which materials have been assessed / what experiments conducted
        :param acquisition_function: The sampling method to be used by the AMI to select new materials
        :param N: The number of materials which the `Greedy N` algorithm is attempting to optimise for
        :param nysamples: Number of samples to draw from posterior for Greedy N optimisation

        :return: ipick: int, the index value in the feature matrix `X` for non-tested materials
        """
        untested = [i for i in range(self.n) if STATUS[i] == 0]
        if acquisition_function == 'Thompson':
            alpha = self.samples()
        elif acquisition_function == 'Greedy_N':
            y_samples = self.samples(nysamples)
            alpha = np.zeros(self.n)
            for j in range(nysamples):
                alpha[np.argpartition(y_samples[:, j], -N)[-N:]] += 1
        else:
            alpha = np.random.rand(self.n)
        ipick = untested[np.argmax(alpha[untested])]
        return ipick
