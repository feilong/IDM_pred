# The neural basis of intelligence in fine-grained cortical topographies

This repository contains the code for our recent work, [The neural basis of intelligence in fine-grained cortical topographies](https://elifesciences.org/articles/64058).

> Feilong, M., Guntupalli, J. S., & Haxby, J. V. (2021). The neural basis of intelligence in fine-grained cortical topographies. eLife, 10, e64058. https://doi.org/10.7554/eLife.64058

In this work, we found that predictions of general intelligence based on fine-grained (vertex-by-vertex) connectivity patterns were markedly stronger than predictions based on coarse-grained (region-by-region) patterns, accounting for approximately twice as much variance. Fine-grained connectivity in the default and frontoparietal cortical systems best predicts intelligence.

> ![Figure 3 adapted](https://github.com/feilong/IDM_pred/raw/main/images/Figure_3_TASK_GitHub.png)
> Comparison of predictions based on fine-grained connectivity and coarse-grained connectivity. Adapted from Figure 3 of the [original paper](https://elifesciences.org/articles/64058).


## Python Environment

The code works with recent versions of Python 3 and its packages. Prior to running the code you need to set up a Python environment using your favorite Python package manager. One way of doing this is to use [conda-forge](https://conda-forge.org/):
```bash
conda create -n IDM_pred 'python>=3.9'
conda activate IDM_pred
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install numpy scipy scikit-learn pandas nibabel joblib ipython jupyter
```

Alternatively the packages can also be installed with [pip](https://pip.pypa.io/en/stable/), preferably with [venv](https://docs.python.org/3/tutorial/venv.html).
```bash
pip install numpy scipy scikit-learn pandas nibabel joblib ipython jupyter
```

The code also uses a package named `IDM_pred` which is included in the repository. Suppose you are in the root directory of a clone of this repository, you can install it in development mode by running:
```bash
cd package/
python setup.py develop
```


## Data

### Subject measures

Two types of data are needed for the analysis. One is subject measures, which can be downloaded from [ConnectomeDB](https://db.humanconnectome.org/data/projects/HCP_1200) as CSV files. After downloading these files, they can be combined into one pickle file using pandas and added to the `IDM_pred` package:
```python
import pandas as pd

# Please replace {FILENAME_1} and {FILENAME_2} with the real file names.
df1 = pd.read_csv('{FILENAME_1}.csv', index_col='Subject')
df2 = pd.read_csv('{FILENAME_2}.csv', index_col='Subject')
df1.index = df1.index.astype(str)
df2.index = df2.index.astype(str)

df = pd.merge(df1, df2, 'outer', left_index=True, right_index=True)
df.to_pickle('package/IDM_pred/io/hcp_full_restricted.pkl')
```
More than 2 CSV files can be combined in a similar manner.

### Connectivity profiles

The other kind of data needed is connectivity profiles. We have condensed these data into individual differences matrix format (subjects x subjects similarity/dissimilarity matrices; Gramian matrices in this case), which can be used to compute the principal components of the connectivity profiles, but takes much less disk space (26 GB) than the original connectivity profiles (terabytes, see figure below on how they were calculated).

> ![Figure 1](https://github.com/feilong/IDM_pred/raw/main/images/Figure_1_conn_profile.png)
> Computing connectivity profiles. Adapted from Figure 1 of the [original paper](https://elifesciences.org/articles/64058).

 We are working on possibilities to share these data openly. In the mean time, you can [contact me](mailto:mafeilong+hcp@gmail.com?subject=HCP_IDM_access) to get a copy provided that you have been granted access to the original HCP dataset.


## Prediction Analysis

> ![Figure 2](https://github.com/feilong/IDM_pred/raw/main/images/Figure_2_schematic_pipeline.png)
> Workflow for the prediction analysis. Adapted from Figure 2 of the [original paper](https://elifesciences.org/articles/64058).

The package `IDM_pred` includes functions that can be used to replicate this analysis. Specifically,
- `IDM_pred.io.get_connectivity_PCs` and `IDM_pred.io.get_measure_info` can be used to load the data.
- `IDM_pred.cv.nested_cv_ridge` implements the ridge-regularized principal components regression model with nested cross-validation.
- `IDM_pred.cv.compute_ss0` computes the sum of squares for null models. It can be used to compute *R<sup>2</sup>*:
    - *R<sup>2</sup>* = 1 - *SS<sub>res</sub>* / *SS<sub>null</sub>*

The script `predict_g.py` is an example to use these functions to replicate the prediction analysis of the paper. To do it, simply run
```bash
python predict_g.py
```

It is highly recommended to run the analysis with a high-performance computing cluster, such as Dartmouth's [Discovery](https://rc.dartmouth.edu/index.php/discovery-overview/).


## Related works

This work has been inspired by many previous works, especially https://github.com/adolphslab/HCP_MRI-behavior and https://github.com/alexhuth/ridge. Please also consider citing these works.
