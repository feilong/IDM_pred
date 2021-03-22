import numpy as np


def split_kfold(families, k=10, seed=None):
    """
    k-fold cross-validation that takes family information into consideration and guarantees that members from the same family are in the same fold.

    Parameters
    ----------
    families : list of ndarrays
        Each element is an ndarray of integers, which are indices for members of a family.
    k : int
        The number of data folds.
    seed : {int, None}

    Returns
    -------
    folds : list of lists
        Each element (fold) is a list, which are families that are in the fold.
    """
    d = {}
    for fam in families:
        l = len(fam)
        if l not in d:
            d[l] = [fam]
        else:
            d[l].append(fam)
    lengths = sorted(d.keys())[::-1]

    rng = np.random.default_rng(seed)

    folds = [[] for _ in range(k)]
    expected_sizes = [len(_) for _ in np.array_split(np.concatenate(families), k)]
    current_sizes = np.zeros((k, ), dtype=int)
    for l in lengths:
        while True:
            s = len(d[l])
            if s == 0:
                break
            fam = d[l].pop(rng.choice(s))
            while True:
                k_ = rng.choice(k)
                if current_sizes[k_] + len(fam) <= expected_sizes[k_]:
                    folds[k_].append(fam)
                    current_sizes[k_] += len(fam)
                    break
    return folds


def split_kfold_simple(families, k=10, seed=None):
    """
    Similar to `split_kfold` but does NOT take family information into consideration. That is, for some test subjects, other members of the family MAY be used for training.
    """
    ss = np.concatenate(families)
    rng = np.random.default_rng(seed)
    rng.shuffle(ss)
    folds = np.array_split(ss, k)
    return folds
