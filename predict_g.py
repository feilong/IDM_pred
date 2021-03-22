import os
from joblib import Parallel, delayed
import numpy as np

from IDM_pred.cv import nested_cv_ridge, compute_ss0
from IDM_pred.io import get_connectivity_PCs, subject_sets, get_measure_info


def prediction_pipeline_region(task, align, info, parcel, y_name, y, mask, families, ss0, overwrite=False):
    out_fn = f'predictions/{y_name}_{task}_{align}_{info}/parcel{parcel:03d}.npz'
    if os.path.exists(out_fn) and not overwrite:
        return
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)

    X = get_connectivity_PCs(task, align, info, parcel, mask=mask)

    yhat = np.zeros_like(y)
    clf_info = np.zeros((len(families), 3))

    for i, fam in enumerate(families):
        yhat[fam], *clf_info[i] = nested_cv_ridge(X, y, fam)

    r2 = 1 - np.sum((y - yhat)**2) / ss0
    np.savez(out_fn, yhat=yhat, clf_info=clf_info, r2=r2)


def prediction_pipeline(y_name, task, align, info, overwrite=False, n_jobs=1):
    """
    Parameters
    ----------
    y_name : str
        Name of the target variable, e.g, `"g"` or `"PMAT24_A_CR"`.
    task : {'TASK', 'REST'}
        The type of fMRI data used to derive the connectivity profiles.
    align : {'ROICHT', 'ROICHR', 'MSMAll'}
        The alignment method applied to the fMRI data. `ROICHT` and `ROICHR` mean ROI Connectivity Hyperalignment based on task and resting data, respectively. `MSMAll` means MSM-aligned data (i.e., no hyperalignment).
    info : {'fine', 'coarse', 'all'}
        The spatial scale of information used. Options are `fine` (residual fine-grained connectivity profiles), `coarse` (coarse-grained connectivity profiles), and `all` (full fine-grained connectivity profiles).
    overwrite: boolean
        Whether to recompute the predictions if the result file already exists.
    n_jobs : int
        The `n_jobs` parameter for joblib's parallel computing.
    """
    y, mask, families, sub_df = get_measure_info(y_name, subject_sets['s888'])
    ss0 = compute_ss0(y, families)

    jobs = []
    for parcel in range(360):
        jobs.append(delayed(prediction_pipeline_region)(
            task, align, info, parcel, y_name, y, mask, families, ss0, overwrite=overwrite
        ))

    if jobs:
        with Parallel(n_jobs=n_jobs, verbose=10, batch_size=1) as parallel:
            parallel(jobs)


if __name__ == '__main__':
    y_name = 'g'
    for task in ['TASK', 'REST']:
        for align in ['ROICHT', 'ROICHR', 'MSMAll']:
            for info in ['fine', 'coarse', 'all']:
                prediction_pipeline(y_name, task, align, info, overwrite=False, n_jobs=32)
