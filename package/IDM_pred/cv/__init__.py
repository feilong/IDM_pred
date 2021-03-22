import numpy as np

from .nested_cv import nested_cv_ridge
from .kfold import split_kfold, split_kfold_simple


def compute_ss0(y, folds):
    """
    Compute the sum of squares based on null models (i.e., always predict the average `y` of the training data).

    Parameters
    ----------
    y : ndarray
    folds : list of ndarray
        Each element is an ndarray of integers, which are the indices of members of the fold.

    Returns
    -------
    ss0 : float
        The sum of squares based on null models.
    """

    yhat0 = np.zeros_like(y)
    for test_idx in folds:
        m = np.ones_like(y, dtype=bool)
        m[test_idx] = False
        yhat0[test_idx] = y[m].mean()
    ss0 = np.sum((y - yhat0)**2)
    return ss0
