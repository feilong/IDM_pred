import os
from glob import glob
import pandas as pd
import numpy as np
from scipy.linalg import eigh

DIR = os.path.dirname(os.path.abspath(__file__))


def get_connectivity_PCs(task, align, info, parcel, ns=888, mask=None,
        IDM_dir='HCP_IDM',
        eps=1e-7,
        ):
    """
    This function outputs a 2-D NumPy array which is the input `X` of the prediction pipeline. The shape of the 2-D array is (n_subjects, n_PCs).
    Each row is a sample, i.e., the PCs of a subject's connectivity profile. Each column is a feature, i.e., a PC's score across all subjects.
    The function loads npy files which contain upper triangles of individual differences matrices (IDMs). See https://github.com/feilong/IDM_pred#data for details.

    Parameters
    ----------
    task : {'TASK', 'REST'}
        The type of fMRI data used to derive the PCs.
    align : {'ROICHT', 'ROICHR', 'MSMAll'}
        The alignment method applied to the fMRI data. `ROICHT` and `ROICHR` mean ROI Connectivity Hyperalignment based on task and resting data, respectively. `MSMAll` means MSM-aligned data (i.e., no hyperalignment).
    info : {'fine', 'coarse', 'all'}
        The spatial scale of information used. Options are `fine` (residual fine-grained connectivity profiles), `coarse` (coarse-grained connectivity profiles), and `all` (full fine-grained connectivity profiles).
    parcel : int
        The number of the parcel whose connectivity profile is used, should be an integer between 0 and 359 (inclusive).
    ns : int, default=888
        The number of subjects in the IDM, default is 888 (the 888 subjects with complete fMRI data).
    mask : array_like or None
        If mask is not None, it should be a boolean array containing `ns` elements. Only subjects whose corresponding value is `True` will be included in the output array.
    IDM_dir : str, default='HCP_IDM'
        The root directory for the IDM data files.
    eps : float, default=1e-7
        The tolerance parameter for eigenvalue decomposition.

    Returns
    -------
    X : ndarray of shape (n_subjects, n_PCs)
    """
    triu_fn = f'{IDM_dir}/{task}/{align}/{info}/parcel{parcel:03d}.npy'
    triu = np.load(triu_fn)
    mat = np.zeros((ns, ns))
    triu_idx = np.triu_indices(ns)
    mat[triu_idx] = triu
    w, v = eigh(mat, lower=False)
    assert np.all(w > -eps)
    w[w < 0] = 0
    U, s = v[:, ::-1][:, :-1], np.sqrt(w[::-1][:-1])
    X = (U * s[np.newaxis])
    if mask is not None:
        X = X[mask]
    return X


def get_df(df_fn=os.path.join(DIR, 'hcp_full_restricted.pkl')):
    """
    This function loads the pandas DataFrame containing subject measures. It should be a pkl file created by `DataFrame.to_pickle`.
    """
    df = pd.read_pickle(df_fn)
    return df


def get_measure_info(y_name, subjects=None):
    """
    Get scores of the target measure for a group of subjects.
    Note that some subjects have invalid data for the measure (e.g., NaNs, missing values), and in such cases only the `n_valid_subjects` subjects with valid data out of the original `n_subjects` subjects are used.

    Parameters
    ----------
    y_name : str
        The name of the target measure, e.g., `"g"` or `"PMAT24_A_CR"`.
    subjects: {list, None}
        The list of subjects used in the analysis, e.g., `["100206"]`. The length of the list is `n_subjects`. `None` means all subjects from the DataFrame.

    Returns
    -------
    y : ndarray of shape (n_valid_subjects, )
    mask : boolean ndarray of shape (n_subjects, )
        The boolean mask, which is `True` for subjects that have valid values of `y` and `False` otherwise. In total there are `n_valid_subjects` `True` values.
    families : list of ndarray
    sub_df : DataFrame of shape (n_valid_subjects, n_measures)
    """
    df = get_df()
    if subjects is not None:
        df = df.loc[subjects]
    y = np.array(df[y_name]).astype(float)

    mask = np.array(np.isfinite(y))
    y = y[mask]

    family_ID = np.array(df['Family_ID'])[mask]
    u = np.unique(family_ID)
    families = [np.where(family_ID == f)[0] for f in u]
    sub_df = df.iloc[np.where(mask)[0]]

    return y, mask, families, sub_df


def _get_subject_sets():
    subject_sets = {}
    for fn in sorted(glob(f'{DIR}/s*.txt')):
        key = os.path.basename(fn)[:-4]
        with open(fn, 'r') as f:
            subjects = f.read().splitlines()
        subject_sets[key] = subjects
    return subject_sets

subject_sets = _get_subject_sets()
