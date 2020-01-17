import pandas as pd
import numpy as np

from pandas.api.types import is_sparse
from scipy.stats import entropy
from sklearn.base import TransformerMixin
from sklearn.utils import check_X_y

from typing import Optional, Union


def pd_entropy(classes: pd.Series) -> float:
    """
    Calculate the class entropy of the subset. The default base of the log is 2, i.e. the amount of information
    needed in bits to specify the classes in the subset.

    Args:
        classes (pd.Series): pandas series of class

    Returns:
        float: the entropy of the class series
    """
    counts = classes.value_counts()
    return entropy(counts, base=2)


def np_entropy(classes: np.ndarray) -> float:
    """
    Calculate the class entropy of the subset. The default base of the log is 2.

    Args:
        classes (np.ndarray): numpy array of the class labels

    Returns:
        float: the entropy of the class labels
    """
    value, counts = np.unique(classes, return_counts=True)
    return entropy(counts, base=2)


# noinspection PyUnresolvedReferences,PyTypeChecker
def pd_information_gain_at_cut_point(x: pd.Series, y: pd.Series, cut_point: float) -> float:
    """
    Calculate the information gain by splitting a numeric attribute at cut_point.

    Args:
        x (pd.Series): pandas series with numerical attributes to split
        y (pd.Series): pandas series of the class labels
        cut_point (float): threshold at which to partition

    Returns:
        float: the information gain by splitting at the cut point
    """
    entropy_before = pd_entropy(y)
    data_left_mask = x <= cut_point
    data_right_mask = x > cut_point
    n_rows = len(x)
    n_left_rows = data_left_mask.sum()
    n_right_rows = data_right_mask.sum()

    gain = (entropy_before -
            (n_left_rows / n_rows) * pd_entropy(y[data_left_mask]) -
            (n_right_rows / n_rows) * pd_entropy(y[data_right_mask]))

    return gain


def np_information_gain_at_cut_point(x: np.ndarray, y: np.ndarray, cut_point: float) -> float:
    """
    Calculate the information gain by splitting a numeric attribute at cut_point.

    Args:
        x (np.ndarray): pandas series with numerical attributes to split
        y (np.ndarray): pandas series of the class labels
        cut_point (float): threshold at which to partition

    Returns:
        float: the information gain by splitting at the cut point
    """
    entropy_before = np_entropy(y)
    data_left_mask = x <= cut_point
    data_right_mask = x > cut_point
    n_rows = len(x)
    n_left_rows = data_left_mask.sum()
    n_right_rows = data_right_mask.sum()

    gain = (entropy_before -
            (n_left_rows / n_rows) * np_entropy(y[data_left_mask]) -
            (n_right_rows / n_rows) * np_entropy(y[data_right_mask]))

    return gain


def get_bad_pandas_dtypes(dtypes: list) -> list:
    # from lightgbm's python-package/lightgbm/basic.py
    pandas_dtype_mapper = {'int8': 'int', 'int16': 'int', 'int32': 'int',
                           'int64': 'int', 'uint8': 'int', 'uint16': 'int',
                           'uint32': 'int', 'uint64': 'int', 'bool': 'int',
                           'float16': 'float', 'float32': 'float', 'float64': 'float'}
    bad_indices = [i for i, dtype in enumerate(dtypes) if (dtype.name not in pandas_dtype_mapper
                                                           and (not is_sparse(dtype)
                                                                or dtype.subtype.name not in pandas_dtype_mapper))]
    return bad_indices


class MDLPDiscretizer(TransformerMixin):
    def __init__(self, features=None):
        """
        This is a sklearn-transformer-compliant MDLP Discretizer that discretizes numeric features according to the
        MDLPC criterion.

        Fayyad and Irani. Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning.
        1993. IJCAI-93.

        Args:
            features: features to discretize (default all)
        """
        self._col_idx = None
        self._bin_descriptions = {}
        self._boundaries = None
        self._ignore_col_idx = None
        self._class_labels = None
        self._n_class = None
        self._data_raw = None
        self._data_raw_np = None
        self._cuts = None

        # Create array with attr indices to discretize
        if features is None:
            # discretize all columns
            self._col_idx = None
        else:
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            # if array[bool, str, int] then (features as mask)
            if (np.issubdtype(features.dtype, np.dtype(int).type) or
                    np.issubdtype(features.dtype, np.dtype(bool).type) or
                    np.issubdtype(features.dtype, np.dtype(str).type)):
                self._col_idx = features
            else:
                raise Exception('features must be np.array of {bool, int, str}')

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        # if array[bool] or array[str], check the length and make sure all columns exist before proceeding further.
        self._data_raw, self._class_labels = check_X_y(X, y)
        self._n_class = np.unique(self._class_labels)
        self.set_col_idx(X)

        # if not all index is given then some must be skipped.
        if len(self._col_idx) != self._data_raw.shape[1]:
            self._ignore_col_idx = np.array([c for c in range(self._data_raw.shape[1]) if c not in self._col_idx])

        # given the col index to discretize, they must all be of acceptable dtype
        if type(X) == pd.DataFrame:
            bad_indices = get_bad_pandas_dtypes(X.iloc[:, self._col_idx].dtypes)
            assert bad_indices == [], 'Did not expect the dtypes in the following column indices: {}'.format(bad_indices)

        # initialize feature bins cut points
        self._cuts = {f: [] for f in self._col_idx}

        # pre-compute all boundary points in dataset
        self._boundaries = self.compute_boundary_points_all_features()

        # get cuts for all features
        self.all_features_accepted_cutpoints()

        # generate bin string descriptions
        self.generate_bin_descriptions()

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        discretized = self.apply_cutpoints(X.copy())
        return discretized

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.DataFrame] = None, **fit_params):
        self.fit(X, y)
        # X may be a pandas dataframe
        _X, _y = check_X_y(X, y)
        return self.transform(_X)

    def set_col_idx(self, X):
        # if features argument was not set, then assume all features need to be discretized
        if self._col_idx is None:
            self._col_idx = np.arange(self._data_raw.shape[1])
        # if array[int], make sure the max index does not exceed the feature shape
        elif np.issubdtype(self._col_idx.dtype, np.dtype(int).type):
            assert np.max(self._col_idx) <= (self._data_raw.shape[1] - 1),\
                'Max index ({}) does not match the feature shape ({})'\
                .format(np.max(self._col_idx), self._data_raw.shape[1])
        # if array[bool], make sure that the length of the mask and features match
        elif np.issubdtype(self._col_idx.dtype, np.dtype(bool).type):
            assert len(self._col_idx) == self._data_raw.shape[1],\
                'The length of the boolean feature mask ({}) must match that of the feature matrix ({}).'\
                .format(len(self._col_idx), self._data_raw.shape[1])
            self._col_idx = np.where(self._col_idx)
        # if array[str], make sure all feature names exist in the feature data frame (assume pandas)
        elif np.issubdtype(self._col_idx.dtype, np.dtype(str).type):
            _col_mask = [c in X.columns for c in X.columns]
            assert np.all(_col_mask), 'These columns do not exist in the data: {}'\
                .format(X.columns[np.invert(_col_mask)])
            self._col_idx = np.array([X.columns.get_loc(c) for c in self._col_idx])
        else:
            raise ValueError

    def mdlpc_criterion(self, X: np.ndarray, y: np.ndarray, cut_point: float) -> bool:
        """
        Determine whether a cut is accepted according to the MDLPC criterion.

        Args:
            X (np.ndarray): feature column
            y (np.ndarray): class column
            cut_point (float): the cut threshold

        Returns:
            bool: whether or not the cut should be accepted. True for accept, False for reject.
        """
        left_mask = X <= cut_point
        right_mask = X > cut_point
        # compute information gain obtained when splitting data at cut_point
        cut_point_gain = np_information_gain_at_cut_point(X, y, cut_point)
        # compute delta term in MDLPC criterion
        n_rows = len(X)  # number of examples in current partition
        partition_entropy = np_entropy(y)
        k = len(np.unique(y))
        k_left = len(np.unique(y[left_mask]))
        k_right = len(np.unique(y[right_mask]))
        entropy_left = np_entropy(y[left_mask])  # entropy of partitioned class
        entropy_right = np_entropy(y[right_mask])
        delta = np.log2(3 ** k) - (k * partition_entropy) + (k_left * entropy_left) + (k_right * entropy_right)

        # to split or not to split
        gain_threshold = (np.log2(n_rows - 1) + delta) / n_rows

        if cut_point_gain > gain_threshold:
            return True
        else:
            return False

    def feature_boundary_points(self, values: np.ndarray) -> np.ndarray:
        """
        Given an attribute, find all potential cut points.

        Args:
            values (np.ndarray): feature values (column)

        Returns:
            np.ndarray: array of potential cut points
        """
        def previous_item(a, val):
            idx = np.where(a == val)[0][0] - 1
            return a[idx]

        missing_mask = np.isnan(values)
        data_partition = np.concatenate([values[:, np.newaxis], self._class_labels[:, np.newaxis]], axis=1)
        data_partition = data_partition[~missing_mask]
        # sort data by values
        data_partition = data_partition[data_partition[:, 0].argsort()]

        # Get unique values in column
        unique_vals = np.unique(data_partition[:, 0])  # each of this could be a bin boundary
        # Find if when feature changes there are different class values
        boundaries = []
        for i in range(1, unique_vals.size):  # By definition first unique value cannot be a boundary
            previous_val_idx = np.where(data_partition[:, 0] == unique_vals[i - 1])[0]
            current_val_idx = np.where(data_partition[:, 0] == unique_vals[i])[0]
            merged_classes = np.union1d(data_partition[previous_val_idx, 1], data_partition[current_val_idx, 1])
            if merged_classes.size > 1:
                boundaries += [unique_vals[i]]
        boundaries_offset = np.array([previous_item(unique_vals, var) for var in boundaries])
        return (np.array(boundaries) + boundaries_offset) / 2

    def compute_boundary_points_all_features(self) -> np.ndarray:
        """
        Compute all possible boundary points for each attribute (column) in self._col_idx.

        Returns:
            np.ndarray: array of boundary points (cut points)

        """
        def padded_cutpoints_array(arr, N):
            cutpoints = self.feature_boundary_points(arr)
            padding = np.array([np.nan] * (N - len(cutpoints)))
            return np.concatenate([cutpoints, padding])

        boundaries = np.empty(self._data_raw.shape)
        boundaries[:, self._col_idx] = np.apply_along_axis(padded_cutpoints_array, 0,
                                                           self._data_raw[:, self._col_idx],
                                                           self._data_raw.shape[0])
        mask = np.all(np.isnan(boundaries), axis=1)
        return boundaries[~mask]

    def boundaries_in_partition(self, X: np.ndarray, feature_idx: int) -> np.ndarray:
        """
        From the collection of all cut points (self._boundaries), find cut points that fall within the value range of
        the feature
        Args:
            X (np.ndarray): feature to split
            feature_idx: index of the feature

        Returns:
            np.ndarray: array of unique cut point values that fall within the value range of the feature
        """
        range_min, range_max = (X.min(), X.max())
        mask = np.logical_and((self._boundaries[:, feature_idx] > range_min),
                              (self._boundaries[:, feature_idx] < range_max))
        return np.unique(self._boundaries[:, feature_idx][mask])

    def best_cut_point(self, X, y, feature_idx) -> Optional[float]:
        """
        Select the best cut point for a feature based on the information gain

        Args:
            X: the feature column
            y: the class label column
            feature_idx: the index of the feature column

        Returns:
            Optional[float, None]: if there are no cut candidates, return None,
            otherwise return the best cut point in float.

        """
        candidates = self.boundaries_in_partition(X, feature_idx=feature_idx)
        if candidates.size == 0:
            return None
        gains = [(cut, np_information_gain_at_cut_point(X, y, cut_point=cut)) for cut in candidates]
        gains = sorted(gains, key=lambda x: x[1], reverse=True)

        return gains[0][0]  # return cut point

    def single_feature_accepted_cutpoints(self, X: np.ndarray, y: np.ndarray, feature_idx) -> None:
        """
        Compute the cuts that are accepted by the MDLPC criterion for a single feature.

        Args:
            X (np.ndarray): single feature column
            y (np.ndarray): class labels
            feature_idx: column index of the feature column

        Returns:
            None

        """
        # delete missing data
        mask = np.isnan(X)
        X = X[~mask]
        y = y[~mask]
        # stop if constant or null feature values
        if len(np.unique(X)) < 2:
            return
        # get cut candidates
        cut_candidate = self.best_cut_point(X, y, feature_idx)
        # return if no candidate is found
        if cut_candidate is None:
            return
        # decide whether to cut
        decision = self.mdlpc_criterion(X, y, cut_candidate)
        if not decision:
            return
        else:
            # partition masks
            left_mask = X <= cut_candidate
            right_mask = X > cut_candidate
            # now we have two new partitions that need to be examined
            left_partition = X[left_mask]
            right_partition = X[right_mask]
            if (left_partition.size == 0) or (right_partition.size == 0):
                return  # extreme point selected, don't partition
            self._cuts[feature_idx] += [cut_candidate]  # accept partition
            self.single_feature_accepted_cutpoints(left_partition, y[left_mask], feature_idx)
            self.single_feature_accepted_cutpoints(right_partition, y[right_mask], feature_idx)
            # order cutpoints in ascending order
            self._cuts[feature_idx] = sorted(self._cuts[feature_idx])
            return

    def all_features_accepted_cutpoints(self):
        """
        Compute cut points that are accepted by the MDLP criterion for all features.

        Returns:
            self
        """
        for attr in self._col_idx:
            self.single_feature_accepted_cutpoints(X=self._data_raw[:, attr], y=self._class_labels, feature_idx=attr)
        return

    def generate_bin_descriptions(self):
        """
        Generate descriptions for the bins.

        Returns:
            self
        """
        bin_label_collection = {}
        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                bin_label_collection[attr] = ['All']
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                start_bin_indices = range(0, len(cuts) - 1)
                bin_labels = ['{}_to_{}'.format(str(cuts[i]), str(cuts[i + 1])) for i in start_bin_indices]
                bin_label_collection[attr] = bin_labels
                self._bin_descriptions[attr] = {i: bin_labels[i] for i in range(len(bin_labels))}

    def apply_cutpoints(self, data: np.ndarray) -> np.ndarray:
        """
        Discretize the data by applying pre-calculated bin conditions.

        Args:
            data: the data to be discretized

        Returns:
            np.ndarray: discretized data
        """
        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                # data[:, attr] = 'All'
                data[:, attr] = 0
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                discretized_col = np.digitize(x=data[:, attr], bins=cuts, right=False).astype('float') - 1
                discretized_col[np.isnan(data[:, attr])] = np.nan
                data[:, attr] = discretized_col
        return data


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    dataset = load_iris()
    org_X, org_y = dataset['data'], dataset['target']
    feature_names, class_names = dataset['feature_names'], dataset['target_names']
    numeric_features = np.arange(org_X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(org_X, org_y, test_size=0.33)

    discretizer = MDLPDiscretizer(features=numeric_features)
    discretizer.fit(X_train, y_train)
    X_train_discretized = discretizer.transform(X_train)
    X_test_discretized = discretizer.transform(X_test)

    print('Original dataset:\n{}'.format(str(X_train[0:5])))
    print('Discretized dataset:\n{}'.format(str(X_train_discretized[0:5])))

    print('Features: {}'.format(', '.join(feature_names)))
    print('Interval cut-points:\n{}'.format(str(discretizer._cuts)))
    print('Bin descriptions:\n{}'.format(str(discretizer._bin_descriptions)))
