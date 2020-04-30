"""
Contains classes used to help the loading and preprocessing of data for screening.
Data can be processed straight from numpy arrays OR through provision of a csv file / mat lab file path.
"""
import pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ami import _checks


class DataTriage(object):
    """Processes feature and target information into formats required by the screening.

    Attributes
    ----------
    X : np.array(), shape(num_entries, num_features)
        Feature matrix.

    y_true : np.array(), shape(num_entries, )
        Target values.

    y_experimental : np.array(), shape(num_entries, )
        Array prepopulated with `Nan` values, to be updated as screening proceeds.

    Methods
    -------
    load_from_path(cls, data_path) --> Not implemented in Parent class.
    """

    def __init__(self, X, y):
        """Checks arrays are not empty, are the same length and have no nan values.

        Parameters
        ----------
        X : np.array(), shape(num_entries, num_features)
            Feature matrix.

        y : np.array(), shape(num_entries, )
            Target values.
        """
        try:
            X, y = np.asarray(X).astype(float), np.asarray(y).astype(float)
        except:
            raise ValueError('Unable to convert target and feature data to float.')

        _checks.array_not_empty(X, y)
        _checks.nan_present(X, y)

        if len(X) != len(y):
            raise ValueError('Lengths of feature and target array do not match')

        self.X = X
        self.y_true, self.y_experimental = self.format_target_values(y)

    @staticmethod
    def _load_dataset_from_path(data_path):
        """Method to be overloaded in further subclasses, loads data for the object.

        Parameters
        ----------
        data_path : str
            Path to file containing data.

        Returns
        -------
        NotImplemented
        """
        return NotImplemented

    @classmethod
    def load_from_path(cls, data_path):
        """Factory method for class when loading data from file.

        Parameters
        ----------
        data_path : str
            Path to file containing data.

        Returns
        -------
        cls(X, y) : DataTriage
            Class method creates DataTriage object.
        """
        _checks.are_type(str, data_path)
        X, y = cls._load_dataset_from_path(data_path)
        return cls(X, y)

    @staticmethod
    def format_target_values(y):
        """Transform target values into array of "known" values and a separate array containing empirical results.
        Empirical array is prepopulated with `nan` values to avoid confusion with target values equal to 0.

        Parameters
        ----------
        y : np.array(), shape(num_entries, )
            Target values.

        Returns
        -------
        (y_true, y_experimental) : (np.array(), np.array()), shape((num_entries, ), (num_entries, ))
        """
        y_true = y
        y_experimental = np.full(len(y), np.nan)
        return y_true, y_experimental


class DataTriageCSV(DataTriage):
    """Child of DataTriage which facilitates loading data from a csv file.
    """

    @staticmethod
    def _load_dataset_from_path(path):
        """Loads data from csv file, assumed target column is rightmost column of the file.
        Uses pandas to load the data so multiple delimiter types are supported.

        Parameters
        ----------
        path : str
            Path to csv file.

        Returns
        -------
        (X, y) : (np.array(), np.array(), shape((num_entries, nun_features), (num_entries, ))
            Feature and target value arrays.
        """
        data = pd.read_csv(path, dtype='float').to_numpy()
        assert data.shape[1] > 1, 'Only one Column present in loaded dataset. X and y must be separate.'
        # check for empty data is performed at initialisation so not duplicating here.
        features, targets = data[:, :-1], data[:, -1]
        return features, targets


class DataTriageMatlab(DataTriage):
    """
    Child class which allows for the loading of data from a matlab file
    """
    @staticmethod
    def _load_dataset_from_path(path):
        """Loads data from matlab file, assumed that feature matrix is located at `X` and target values at `y`.

        Parameters
        ----------
        path : str
            Path to matlab file.

        Returns
        -------
        (X, y) : (np.array(), np.array(), shape((num_entries, nun_features), (num_entries, ))
            Feature and target value arrays.
        """
        data_set = loadmat(path, appendmat=False)
        features, targets = data_set['X'], data_set['y'].ravel()  # ravel due to loading as column vector
        return features, targets


class DataTriagePickle(DataTriage):
    """
    Child class which allows for the loading of data from a pickle file
    """
    @staticmethod
    def _load_dataset_from_path(path):
        """Loads data from pickle file, assumed that feature matrix is located at `X` and target values at `y`.

        Parameters
        ----------
        path : str
            Path to pickle file.

        Returns
        -------
        (X, y) : (np.array(), np.array(), shape((num_entries, nun_features), (num_entries, ))
            Feature and target value arrays.
        """
        with open(path, 'rb') as f:
            data_set = pickle.load(f)
        features, targets = data_set['X'], data_set['y']
        return features, targets
