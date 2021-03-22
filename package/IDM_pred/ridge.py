import numpy as np
from scipy.linalg import svd, LinAlgError


def ridge(X_train, X_test, y_train, alpha, npc, fit_intercept=True):
    """
    Parameters
    ----------
    X_train : ndarray of shape (n_training_samples, n_features)
    X_test : ndarray of shape (n_test_samples, n_features)
    y_train : ndarray of shape (n_training_samples, )
    alpha : float
        The regularization parameter for ridge regression.
    npc : {int, None}
        The number of PCs used in the prediction model. `None` means all PCs are used.

    Returns
    -------
    yhat : ndarray of shape (n_test_samples, )
        Predicted values of the target variable.
    """
    # TODO multiple measures
    if fit_intercept:
        X_offset = np.mean(X_train, axis=0, keepdims=True)
        y_offset = np.mean(y_train, axis=0, keepdims=True)
        X_train = X_train - X_offset
        y_train = y_train - y_offset
    try:
        U, s, Vt = svd(X_train, full_matrices=False)
    except LinAlgError:
        U, s, Vt = svd(X_train, full_matrices=False, lapack_driver='gesvd')
    if fit_intercept:
        X_test_V = (X_test - X_offset) @ Vt.T[:, :npc]
    else:
        X_test_V = X_test @ Vt.T[:, :npc]
    UT_y = U.T[:npc, :] @ y_train
    d = s[:npc] / (s[:npc]**2 + alpha)
    d_UT_y = d * UT_y
    yhat = (X_test_V @ d_UT_y)
    if fit_intercept:
        yhat += y_offset
    return yhat


def grid_ridge(X_train, X_test, y_train, alphas, npcs, fit_intercept=True):
    """
    Parameters
    ----------
    X_train : ndarray of shape (n_training_samples, n_features)
    X_test : ndarray of shape (n_test_samples, n_features)
    y_train : ndarray of shape (n_training_samples, )
    alphas : {list, ndarray of shape (n_alphas, )}
        Choices of the regularization parameter for ridge regression.
    npcs : {list, ndarray of shape (n_npcs, )}
        Choices of the number of PCs used in the prediction model in increasing order. Each element in the list should be an integer or `None`. `None` means all PCs are used.

    Returns
    -------
    yhat : ndarray of shape (n_test_samples, n_alphas, n_npcs)
    """
    if fit_intercept:
        X_offset = np.mean(X_train, axis=0, keepdims=True)
        y_offset = np.mean(y_train, axis=0, keepdims=True)
        X_train = X_train - X_offset
        y_train = y_train - y_offset
    try:
        U, s, Vt = svd(X_train, full_matrices=False)
    except LinAlgError:
        U, s, Vt = svd(X_train, full_matrices=False, lapack_driver='gesvd')

    if fit_intercept:
        X_test_V = (X_test - X_offset) @ Vt.T  # (n_test_samples, k)
    else:
        X_test_V = X_test @ Vt.T
    UT_y = U.T @ y_train  # (k, )

    d = s[:, np.newaxis] / ((s**2)[:, np.newaxis] + np.array(alphas)[np.newaxis, :])  # (k, n_alphas)
    if len(y_train.shape) == 1:
        d_UT_y = d * UT_y[..., np.newaxis]  # (k, n_alphas)
        yhat = np.zeros((X_test.shape[0], len(alphas), len(npcs)))
    else:
        d_UT_y = d[:, np.newaxis, :] * UT_y[..., np.newaxis]  # (k, n_measures, n_alphas)
        yhat = np.zeros((X_test.shape[0], y_train.shape[1], len(alphas), len(npcs)))
        y_offset = y_offset[:, :, np.newaxis, np.newaxis]

    npcs_ = [0] + list(npcs)
    for i in range(len(npcs)):
        yhat[..., i] = np.tensordot(X_test_V[:, npcs_[i]:npcs[i]], d_UT_y[npcs_[i]:npcs[i]], axes=(1, 0))
    yhat = np.cumsum(yhat, axis=-1)
    if fit_intercept:
        yhat += y_offset
    return yhat
