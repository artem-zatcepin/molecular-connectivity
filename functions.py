import numpy as np
import pandas as pd
import numba
import warnings


def extract(arr, atlas, label=1, feature='mean', percent_hottest=0.2):
    inds = atlas.mask[label]
    try:
        extraction_function = getattr(np, feature)
        return extraction_function(arr[inds])

    except AttributeError:
        if feature == 'hottest_mean':
            arr_flat = arr[inds]
            n_hottest = int(percent_hottest * len(arr_flat))
            return np.mean(np.sort(arr_flat)[::-1][:n_hottest])
        else:
            raise NotImplementedError('Fallback to PyRadiomics is not implemented yet. Choose a valid numpy function')

@numba.njit
def linreg(x, y):
    A = np.vstack((x, np.ones(len(x)))).T
    return np.linalg.lstsq(A, y, rcond=-1.0)[0]

def lists_identical(list1, list2):
    return all(x == y for x, y in zip(list1, list2))

def distance_matrix(features, K, B):
    # input:
    # features - dataframe of extracted features (e.g., extracted VOI means) for test subjects, subjects are in index, VOIs are in columns
    # K - dataframe of slopes calculated for a normal cohort
    # B - dataframe of intercepts calculated for a normal cohort
    # returns:
    # Ds - list of distance matrices for test subjects

    sbjs = features.index
    Ds = {}
    for sbj in sbjs:
        sbj_features = features.loc[sbj]
        D = pd.DataFrame(index=K.index, columns=K.columns)
        for voi1 in K.index:
            for voi2 in K.columns:
                if voi1 == voi2:
                    continue
                k = K.loc[voi1, voi2]
                b = B.loc[voi1, voi2]
                if k == 0 and b == 0:
                    continue
                d_x = np.abs(k * sbj_features[voi1] + b - sbj_features[voi2])
                d_y = np.abs((sbj_features[voi2] - b) / k - sbj_features[voi1])
                D.loc[voi1, voi2] = d_x * d_y / np.sqrt(d_x * d_x + d_y * d_y)
        Ds[sbj] = D
    return Ds

def cds_per_voi(features, K, B):
    Ds = distance_matrix(features, K, B)
    df_sum_Ds = pd.DataFrame(index=features.index, columns=K.index, dtype=float)
    for key, D in Ds.items():
        df_sum_Ds.loc[key] = D.sum(axis=1, skipna=True, min_count=1)
    return df_sum_Ds
