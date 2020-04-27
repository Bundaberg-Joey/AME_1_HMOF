
import pytest
import numpy as np

from ami.data import DataTriage, DataTriageCSV, DataTriageMatlab, DataTriagePickle


# DataTriage -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("X, y",
                         [
                             (np.random.randn(1000, 20), np.random.randn(1000)),
                             (['0.01', '0.02', '0.03'], ['1.0', '2.0', '3.0'])  # convert to float
                         ])
def test_datatriage_pass(X, y):
    data = DataTriage(X, y)
    assert data.y_true.shape == data.y_experimental.shape  # target arrays have same shape
    assert data.y_true.ndim == 1  # flat array
    pass


@pytest.mark.xfail(reason='Invalid arguments')
@pytest.mark.parametrize("X, y",
                         [
                             (['a', '0.02', '0.03'], ['a', '2', '3']),  # must convert to float
                             (np.random.randn(2, 10), []),  # empty array
                             ([], np.random.randn(5)),  # emptry feature
                             ([np.nan, 5, 2, 3], [1, 2, 3, 4]),  # nan present in feature
                             ([1, 2, 3, 4], [np.nan, 2, 3, 4]),  # nan present in target
                             ([1, 2, 3], [1])  # arrays are different lengths
                         ])
def test_datatriage_fail(X, y):
    data = DataTriage(X, y)


# DataTriage Factory methods -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("loader, path, x_dim, y_dim",
                         [
                             (DataTriageCSV, 'tests/files/_mock_data_pass.csv', (1000, 20), (1000, )),
                             (DataTriageMatlab, 'tests/files/_mock_data_pass.mat', (1000, 20), (1000, )),
                             (DataTriagePickle, 'tests/files/_mock_data_pass.pkl', (1000, 20), (1000, ))
                         ])
def test_datatriage_factory_pass(path, loader, x_dim, y_dim):
    data = loader.load_from_path(path)
    X, y = data.X, data.y_true
    assert X.shape == x_dim # loaded the data in the correct shape
    assert y.shape == y_dim
    assert X.dtype == y.dtype == 'float'  # loaded data is float type


@pytest.mark.xfail(reason='Invalid file paths and internal class checks')
@pytest.mark.parametrize("loader, path",
                         [
                             (DataTriageCSV, 'fake/path/rand.csv'),
                             (DataTriageMatlab, 'fake/path/rand.mat'),
                             (DataTriagePickle, 'fake/path/rand.pkl'),
                             (DataTriageCSV, 'tests/files/_mock_data_1col.csv'),
                             (DataTriageMatlab, 'tests/files/_mock_data_labels.mat'),
                             (DataTriagePickle, 'tests/files/_mock_data_labels.pkl')

                         ])
def test_datatriage_factory_fail(loader, path):
    data = loader.load_from_path(path)


# ----------------------------------------------------------------------------------------------------------------------
