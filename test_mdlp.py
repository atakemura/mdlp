import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mdlp import MDLPDiscretizer


@pytest.fixture(scope='module')
def iris_data():
    dataset = load_iris()
    org_X, org_y = dataset['data'], dataset['target']
    feature_names, class_names = dataset['feature_names'], dataset['target_names']
    # numeric_features = np.arange(org_X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(org_X, org_y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test, feature_names, class_names


class TestMDLPDiscretizer:
    def test_iris_all_numeric(self, iris_data):
        X_train, X_test, y_train, y_test, feature_names, class_names = iris_data
        discretizer = MDLPDiscretizer(return_df=True)
        discretizer.fit(X_train, y_train)
        X_train_discretized = discretizer.transform(X_train)
        X_test_discretized = discretizer.transform(X_test)

        print('Original dataset:\n{}'.format(str(X_train[0:5])))
        print('Discretized dataset:\n{}'.format(str(X_train_discretized[0:5])))

        print('Features: {}'.format(', '.join(feature_names)))
        print('Interval cut-points:\n{}'.format(str(discretizer.cuts)))
        print('Bin descriptions:\n{}'.format(str(discretizer.bin_descriptions)))

        assert type(X_train_discretized) == pd.DataFrame
        assert type(X_test_discretized) == pd.DataFrame

    def test_iris_mixed_string(self, iris_data):
        X_train, X_test, y_train, y_test, feature_names, class_names = iris_data
        X_train, y_train = pd.DataFrame(X_train, columns=feature_names), pd.Series(y_train)
        X_test, y_test = pd.DataFrame(X_test, columns=feature_names), pd.Series(y_test)

        X_train.iloc[:, 3] = X_train.iloc[:, 3].astype('str')
        X_test.iloc[:, 3] = X_test.iloc[:, 3].astype('str')

        discretizer = MDLPDiscretizer(features=feature_names[:3])
        discretizer.fit(X_train, y_train)
        X_train_discretized = discretizer.transform(X_train)
        X_test_discretized = discretizer.transform(X_test)

        assert type(X_train_discretized) == pd.DataFrame
        assert type(X_test_discretized) == pd.DataFrame

        assert X_train_discretized.shape == X_train.shape
        assert X_test_discretized.shape == X_test.shape

        assert discretizer.bin_descriptions == {0: {0: '-inf_to_5.55', 1: '5.55_to_inf'},
                                                1: {0: '-inf_to_3.25', 1: '3.25_to_inf'},
                                                2: {0: '-inf_to_2.45', 1: '2.45_to_4.75', 2: '4.75_to_inf'}}

    def test_iris_return_numpy(self, iris_data):
        X_train, X_test, y_train, y_test, feature_names, class_names = iris_data
        X_train, y_train = pd.DataFrame(X_train, columns=feature_names), pd.Series(y_train)
        X_test, y_test = pd.DataFrame(X_test, columns=feature_names), pd.Series(y_test)

        X_train.iloc[:, 3] = X_train.iloc[:, 3].astype('str')
        X_test.iloc[:, 3] = X_test.iloc[:, 3].astype('str')

        discretizer = MDLPDiscretizer(features=feature_names[:3], return_df=False)
        discretizer.fit(X_train, y_train)
        X_train_discretized = discretizer.transform(X_train)
        X_test_discretized = discretizer.transform(X_test)

        assert type(X_train_discretized) == np.ndarray
        assert type(X_test_discretized) == np.ndarray

        assert X_train_discretized.shape == X_train.shape
        assert X_test_discretized.shape == X_test.shape

        assert discretizer.bin_descriptions == {0: {0: '-inf_to_5.55', 1: '5.55_to_inf'},
                                                1: {0: '-inf_to_3.25', 1: '3.25_to_inf'},
                                                2: {0: '-inf_to_2.45', 1: '2.45_to_4.75', 2: '4.75_to_inf'}}


    def test_iris_return_numpy_categorical_codes(self, iris_data):
        X_train, X_test, y_train, y_test, feature_names, class_names = iris_data
        X_train, y_train = pd.DataFrame(X_train, columns=feature_names), pd.Series(y_train)
        X_test, y_test = pd.DataFrame(X_test, columns=feature_names), pd.Series(y_test)

        X_train.iloc[:, 3] = X_train.iloc[:, 3].astype('str')
        X_test.iloc[:, 3] = X_test.iloc[:, 3].astype('str')

        discretizer = MDLPDiscretizer(features=feature_names[:3], return_df=False, return_intervals=False)
        discretizer.fit(X_train, y_train)
        X_train_discretized = discretizer.transform(X_train)
        X_test_discretized = discretizer.transform(X_test)

        assert type(X_train_discretized) == np.ndarray
        assert type(X_test_discretized) == np.ndarray

        assert X_train_discretized.shape == X_train.shape
        assert X_test_discretized.shape == X_test.shape

        assert all(np.unique(X_train_discretized[:, 0]) == np.array([0,1]))
        assert all(np.unique(X_train_discretized[:, 1]) == np.array([0,1]))
        assert all(np.unique(X_train_discretized[:, 2]) == np.array([0,1,2]))
