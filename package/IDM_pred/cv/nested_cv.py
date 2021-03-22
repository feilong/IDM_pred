import numpy as np
from sklearn.model_selection import StratifiedKFold

from IDM_pred.ridge import ridge, grid_ridge


def nested_cv_ridge(
        X, y, test_index, n_bins=4, n_folds=3,
        alphas = 10**np.linspace(-20, 20, 81),
        npcs=[10, 20, 40, 80, 160, 320, None],
        train_index=None,
    ):
    """
    Predict the scores of the testing subjects based on data from the training subjects using ridge regression. Hyperparameters are chosen based on a nested cross-validation. The inner-loop of the nested cross-validation is a stratified k-fold cross-validation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples, )
    test_idx : ndarray of shape (n_test_samples, )
        Indices for the samples that are used for testing.
    n_bins : int
        Training data are divided into `n_bins` bins for stratified k-fold cross-validation.
    n_folds : int
        Number of folds for stratified k-fold cross-validation.
    alphas : {list, ndarray of shape (n_alphas, )}
        Choices of the regularization parameter for ridge regression.
    npcs : list
        Choices of the number of PCs used in the prediction model in increasing order. Each element in the list should be an integer or `None`. `None` means all PCs are used.
    train_idx : {None, ndarray of shape (n_training_samples, )}
        Indices for the samples that are used for training. If it is `None`, then all the samples except for the test samples are used.

    Returns
    -------
    yhat : ndarray of shape (n_test_samples, )
        Predicted scores for the test samples.
    alpha : float
        The chosen element of `alphas` based on nested cross-validation.
    npc : {int, None}
        The chosen element of `npcs` based on nested cross-validation.
    cost : float
        The cost based on the chosen hyperparameters, which is the minimum cost for training data among all hyperparameter choices. 
    """
    if train_index is None:
        train_index = np.setdiff1d(np.arange(X.shape[0], dtype=int), test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    bin_limits = np.histogram(y_train, n_bins)[1]
    bins = np.digitize(y_train, bin_limits[:-1])

    cv = StratifiedKFold(n_splits=n_folds)
    costs = []
    for train, test in cv.split(X_train, bins):
        yhat = grid_ridge(X_train[train], X_train[test], y_train[train], alphas, npcs)
        cost = ((y_train[test][:, np.newaxis, np.newaxis] - yhat)**2).sum(axis=0)
        costs.append(cost)
    costs = np.sum(costs, axis=0)
    a, b = np.unravel_index(costs.argmin(), costs.shape)

    alpha = alphas[a]
    npc = npcs[b]

    yhat = ridge(X_train, X_test, y_train, alpha, npc)

    return yhat, alpha, npc, costs[a, b]
