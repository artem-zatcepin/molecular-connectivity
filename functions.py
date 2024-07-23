import numpy as np
import pandas as pd
import nibabel as nib
import pingouin as pg
import numba
import warnings


def extract(arr, atlas, label=1, feature='mean', percent_hottest=0.2, n_hottest=None):
    inds = atlas.mask[label]
    try:
        extraction_function = getattr(np, feature)
        return extraction_function(arr[inds])

    except AttributeError:
        if feature == 'hottest_mean':
            arr_flat = arr[inds]
            if n_hottest is None:
                n_hottest = int(percent_hottest * len(arr_flat))
            return np.mean(np.sort(arr_flat)[::-1][:n_hottest])
        else:
            raise NotImplementedError('Fallback to PyRadiomics is not implemented yet. Choose a valid numpy function')

@numba.njit
def linreg(x, y):
    A = np.vstack((x, np.ones(len(x)))).T
    return np.linalg.lstsq(A, y, rcond=-1.0)[0]

@np.errstate(invalid='raise', divide='raise')
def get_correlation_matrix(data: pd.DataFrame,
                           cov_df=None,
                           kind='correlation',
                           method='pearson',
                           ):
    if cov_df is None:
        full_df = data
    else:
        cov_df = cov_df.drop(get_cols_with_zero_variance(cov_df), axis='columns')  # in cases when e.g. the sex of all subjects is the same, the variance would be 0, and the correlation couldn't be calculated
        full_df = pd.concat([data, cov_df], axis=1)

    if kind == 'correlation':
        if cov_df is None:
            return data.corr(method).values
        else:
            warnings.warn('Warning: current implementation of full correlation matrix calculation with covariates '
                          'is very slow. Consider reducing N bootstraps')
            out_df = pd.DataFrame(index=data.columns, columns=data.columns)
            for i, col in enumerate(data.columns):
                for row in data.columns[i+1:]:
                    #out_df.loc[row, col] = pg.partial_corr(data=full_df, x=row, y=col, covar=list(cov_df.columns), method=method)['r'].iloc[0]
                    cols_temp = [row, col] + list(cov_df.columns)
                    try:
                        df_temp = full_df[cols_temp].copy().astype(np.float64).round(15)
                    except FloatingPointError:
                        df_temp = full_df[cols_temp].copy().astype(np.float64).round(14)
                    out_df.loc[row, col] = df_temp.pcorr().loc[row, col]
            out_arr = np.tril(out_df.values)
            out_arr += out_arr.T
            return out_arr.astype(float)

    elif kind == 'partial correlation':
        # we cannot obtain p-values using pg.partial_corr since n < k in our case
        # where n is sample size, k is number of covariates, so dof = n - k - 2 < 0 and
        # we get a negative value under the sqrt (see pg.partial_corr)
        if method != 'pearson':
            raise NotImplementedError("For partial correlations (partial correlations for each pair given all others),"
                                      " only method='pearson' is implemented")
        try:
            return full_df.astype(np.float64).round(15).pcorr().loc[data.columns, data.columns].values
        except FloatingPointError:
            return full_df.astype(np.float64).round(14).pcorr().loc[data.columns, data.columns].values

    else:
        raise Exception(f'{kind} cannot be calculated for the selected estimator')

def check_for_infs_in_matrix(matrix):  # matrix is a 3D numpy array (3rd dimension represents individual bootstraps)
    inds_neg_inf = np.where(matrix == -np.inf)
    inds_pos_inf = np.where(matrix == np.inf)
    if len(inds_neg_inf[0]) > 0:
        matrix[inds_neg_inf] = -10
        warnings.warn("Warning: Fisher's Z is -inf in at least one of the bootstraps. Replacing with -10. Your sample might be too small.")
    if len(inds_pos_inf[0]) > 0:
        matrix[inds_pos_inf] = 10
        warnings.warn("Warning: Fisher's Z is inf in at least one of the bootstraps. Replacing with 10. Your sample might be too small.")
    return matrix

def get_cols_with_zero_variance(df: pd.DataFrame):
    zero_variance_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            zero_variance_cols.append(col)
    return zero_variance_cols

def reorient_to_target(input_nifti, target_nifti, set_target_affine=False, verbose=False):
    target_ornt = nib.orientations.io_orientation(target_nifti.affine)
    if verbose: print(f'Target orientation: {nib.orientations.aff2axcodes(target_nifti.affine)}')
    input_ornt = nib.orientations.io_orientation(input_nifti.affine)
    transform = nib.orientations.ornt_transform(input_ornt, target_ornt)
    if verbose: print(f'Input image with orientation {nib.orientations.aff2axcodes(input_nifti.affine)} is reoriented to target')
    out_nifti = input_nifti.as_reoriented(transform)
    if set_target_affine:
        out_nifti = nib.Nifti1Image(dataobj=input_nifti.dataobj, affine=target_nifti.affine, header=input_nifti.header)
    return out_nifti

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
        for i, voi1 in enumerate(K.index):
            for j in range(i + 1, len(K.columns)):
                voi2 = K.columns[j]
                k = K.loc[voi1, voi2]
                b = B.loc[voi1, voi2]
                if k == 0 and b == 0:
                    continue
                d_x = np.abs(k * sbj_features[voi1] + b - sbj_features[voi2])
                d_y = np.abs((sbj_features[voi2] - b) / k - sbj_features[voi1])
                D.loc[voi1, voi2] = D.loc[voi2, voi1] = d_x * d_y / np.sqrt(d_x * d_x + d_y * d_y)
        Ds[sbj] = D
    return Ds

def cds_per_voi(features, K, B):
    Ds = distance_matrix(features, K, B)
    df_sum_Ds = pd.DataFrame(index=features.index, columns=K.index, dtype=float)
    for key, D in Ds.items():
        df_sum_Ds.loc[key] = D.sum(axis=1, skipna=True, min_count=1)
    return df_sum_Ds
