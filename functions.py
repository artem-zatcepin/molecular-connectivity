import numpy as np
import pandas as pd
import nibabel as nib
import pingouin as pg
import numba
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
import re


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
                           new_partial=True,
                           ):
    if cov_df is None:
        full_df = data
    else:
        cov_df = cov_df.drop(get_cols_with_zero_variance(cov_df), axis='columns')  # in cases when e.g. the sex of all subjects is the same, the variance would be 0, and the correlation couldn't be calculated
        full_df = pd.concat([data, cov_df], axis=1)

    if kind == 'correlation':
        if cov_df is None:
            return data.corr(method).values

        elif new_partial:
            df_resid = data.copy()
            if len(cov_df.columns) != 0:
                for col in data.columns:
                    y = data[col].values
                    X = cov_df.values
                    reg = LinearRegression().fit(X, y)
                    df_resid[col] = y - reg.predict(X)
            return df_resid.corr(method).values

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

def pairwise_cds_distr_params(features=None, K=None, B=None, Ds=None):
    if Ds is None:
        Ds = distance_matrix(features, K, B)
    out_dict = {}

    Ds_list = []
    for D in Ds.values():
        np.fill_diagonal(D.to_numpy(), 0)
        Ds_list.append(D)

    Ds_arr = np.array(Ds_list)

    out_dict['distance_matrix'] = Ds
    out_dict['median'] = np.median(Ds_arr, axis=0)
    out_dict['mean'] = np.mean(Ds_arr, axis=0)
    #out_dict['std'] = np.std(Ds_arr, axis=0)
    out_dict['percentile_25'] = np.percentile(Ds_arr, 25, axis=0)
    out_dict['percentile_75'] = np.percentile(Ds_arr, 75, axis=0)
    return out_dict

def cds_per_voi(features, K, B):
    Ds = distance_matrix(features, K, B)
    df_sum_Ds = pd.DataFrame(index=features.index, columns=K.index, dtype=float)
    for key, D in Ds.items():
        df_sum_Ds.loc[key] = D.sum(axis=1, skipna=True, min_count=1)
    return df_sum_Ds


def sum_cds_matrix_per_cohort(features, K, B):
    Ds = distance_matrix(features, K, B)

    Ds_arr = np.array([D.to_numpy() for D in Ds.values()])
    Ds_cohort_sum = np.nansum(Ds_arr, axis=0)

    return Ds_cohort_sum

# symmetric kullback-leibler divergence
def symmetric_kl(pdf_p, pdf_q, sample_points):

    p = pdf_p(sample_points)
    q = pdf_q(sample_points)

    # Ensure no zero values for caluclation
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)

    return stats.entropy(p, q) + stats.entropy(q, p)

def compute_adjacency_matrix(df, gene_col='value', clusters_col='clusters', n_sample_points=1000):
    # Input
    # df: gene expression dataframe containing cluster information
    # gene_col: column containing expression levels of the gene/score of interest
    # clusters_col: column containing cluster labels

    # Returns
    # adjacency matrix (np.ndarray)

    clusters = list(df[clusters_col].unique())
    n_clusters = len(clusters)
    adjacency_matrix = np.zeros((n_clusters, n_clusters))

    # Generate sample points for KDE evaluation
    sample_points = np.linspace(df[gene_col].min(), df[gene_col].max(), n_sample_points)
    for i, cluster1 in enumerate(clusters):
        values_1 = df.loc[df[clusters_col] == cluster1, gene_col].to_numpy()
        pdf_1 = stats.gaussian_kde(values_1)

        for j, cluster2 in enumerate(clusters):
            values_2 = df.loc[df[clusters_col] == cluster2, gene_col].to_numpy()
            pdf_2 = stats.gaussian_kde(values_2)

            kld = symmetric_kl(pdf_1, pdf_2, sample_points)
            similarity = np.exp(-kld)

            adjacency_matrix[i, j] = similarity

    return adjacency_matrix


def distribution_info(df):
    results = {
        'Shapiro p-value': {},  # Specifies p-value comes from Shapiro-Wilk test
        'is_normal': {},  # Boolean for normality check
        'mean': {},  # Mean of the distribution
        'median': {},  # Median of the distribution
        'IQR_25': {},  # 25th percentile (Q1)
        'IQR_75': {}  # 75th percentile (Q3)
    }

    for col in df.select_dtypes(include=['number']):
        data = df[col].dropna()  # Drop NaN values for accuracy
        p_value = stats.shapiro(data).pvalue  # Shapiro-Wilk normality test

        # Store results
        results['Shapiro p-value'][col] = p_value
        results['is_normal'][col] = p_value > 0.05  # Normal if p > 0.05
        results['mean'][col] = data.mean()
        results['median'][col] = data.median()
        results['IQR_25'][col] = data.quantile(0.25)  # Q1 (25th percentile)
        results['IQR_75'][col] = data.quantile(0.75)  # Q3 (75th percentile)

    return pd.DataFrame(results)


def add_percent_difference(distribution_info_dict, ref_cohort_key):
    """
    Adds percent difference columns for mean and median relative to a reference cohort.

    Parameters:
        - distribution_info_dict (dict): Dictionary of DataFrames with cohort names as keys.
        - ref_cohort_key (str): The key of the reference cohort.

    Returns:
        - dict: Updated dictionary with added percent difference columns.
    """

    if ref_cohort_key not in distribution_info_dict:
        raise ValueError(f"Reference cohort '{ref_cohort_key}' not found in the dictionary.")

    ref_df = distribution_info_dict[ref_cohort_key]  # Reference cohort DataFrame

    for cohort, df in distribution_info_dict.items():
        if cohort == ref_cohort_key:
            continue  # Skip reference cohort

        updated_df = df.copy()  # Copy to avoid modifying original

        # Compute percent difference for mean and median
        updated_df[f'% diff mean ({ref_cohort_key})'] = (
                (df['mean'] - ref_df['mean']) / ref_df['mean'] * 100
        ).fillna(0)  # Fill NaNs (if division by zero occurs)

        updated_df[f'% diff median ({ref_cohort_key})'] = (
                (df['median'] - ref_df['median']) / ref_df['median'] * 100
        ).fillna(0)  # Fill NaNs

        distribution_info_dict[cohort] = updated_df  # Update the dictionary

    return distribution_info_dict


def sanitize_filename(filename, replacement="_"):
    # Windows disallowed characters for filenames
    forbidden_chars = r'[<>:"/\\|?*]'

    # Trim trailing dots and spaces (Windows does not allow them)
    filename = re.sub(forbidden_chars, replacement, filename).strip().rstrip(".")

    return filename







