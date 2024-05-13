from functions import linreg, get_correlation_matrix, check_for_infs_in_matrix
import warnings

import numpy as np
import pandas as pd
import pingouin as pg

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV
import sklearn.linear_model
from nilearn.connectome import ConnectivityMeasure


def init_network(df: pd.DataFrame,
                 network1_name='',
                 network2_name=None,
                 ):
    if network1_name == network2_name or network2_name is None:
        return Network(df, name=network1_name)
    else:
        return BetweenNetwork(df, network1_name=network1_name, network2_name=network2_name)


class NetworkBase:
    def __init__(self,
                 df: pd.DataFrame,  # connectivity matrix, index and columns are vois
                 df_K=None,  # matrix of slopes, optional
                 df_B=None,  # matrix of intercepts, optional
                 name='',
                 ):

        self.df = df
        self.df_no_duplicates = None
        self.matrix = df.to_numpy()
        self.matrix_binary = self.matrix != 0
        self.name = name
        self.vois_rows = df.index.to_list()
        self.vois_cols = df.columns.to_list()
        if df_K is not None:
            self.df_K = df_K
        if df_B is not None:
            self.df_B = df_B
    def __str__(self):
        return self.name
    def compute_distribution_params(self):
        self.df_valid_long = pd.melt(self.df_no_duplicates, var_name='VOI_2', ignore_index=False)
        self.df_valid_long = self.df_valid_long.loc[self.df_valid_long['value'] != 0]
        self.df_valid_long.reset_index(names='VOI_1', inplace=True)
        self.df_valid_long['abs_value'] = self.df_valid_long['value'].abs()
        self.df_valid_long['network_name'] = [self.name] * len(self.df_valid_long)

        self.df_valid_long = self.df_valid_long[['network_name', 'VOI_1', 'VOI_2', 'value', 'abs_value']]
        self.df_valid_long['VOI_pair'] = [f'{voi1}_&_{voi2}' for voi1, voi2 in zip(self.df_valid_long['VOI_1'], self.df_valid_long['VOI_2'])]
        matrix_flat = self.df_valid_long['value'].to_numpy()
        matrix_flat_abs = self.df_valid_long['abs_value'].to_numpy()

        try:
            self.mean = np.mean(matrix_flat)
            self.median = np.median(matrix_flat)
            self.std = np.std(matrix_flat)
            self.min = np.min(matrix_flat)
            self.max = np.max(matrix_flat)

            self.abs_mean = np.mean(matrix_flat_abs)
            self.abs_median = np.median(matrix_flat_abs)
            self.abs_std = np.std(matrix_flat_abs)
            self.abs_min = np.min(matrix_flat_abs)
            self.abs_max = np.max(matrix_flat_abs)
        except ValueError:
            self.min = 0
            self.max = 0
            print(f'WARNING: Network {self.name} has no connections')

class Network(NetworkBase):
    def __init__(self,
                 df: pd.DataFrame,
                 df_K=None,
                 df_B=None,
                 name='',
                 ):
        super().__init__(df, df_K=df_K, df_B=df_B, name=name)
        assert not np.any(np.diag(self.matrix)), 'At least one of diagonal elements is non-zero'
        self.vois = self.vois_rows
        self.df_no_duplicates = pd.DataFrame(np.tril(self.matrix), index=self.vois_rows, columns=self.vois_cols)
        self.n_connect = np.sum(np.tril(self.matrix_binary))
        self.n_connect_per_voi = dict(zip(self.vois, np.sum(self.matrix_binary, axis=0)))
        self.compute_distribution_params()


class BetweenNetwork(NetworkBase):
    def __init__(self,
                 df: pd.DataFrame,
                 df_K=None,
                 df_B=None,
                 network1_name='',
                 network2_name='',
                 ):
        self.net1_name = network1_name
        self.net2_name = network2_name
        name = f'{network1_name}-{network2_name}'

        super().__init__(df, df_K=df_K, df_B=df_B, name=name)

        self.df_no_duplicates = self.df  # rows and cols are distinct, so no duplicates in df
        self.n_connect = np.sum(self.matrix_binary)
        n_connect_per_voi_x = dict(zip(self.vois_rows, np.sum(self.matrix_binary, axis=1)))
        n_connect_per_voi_y = dict(zip(self.vois_cols, np.sum(self.matrix_binary, axis=0)))
        self.n_connect_per_voi = n_connect_per_voi_x | n_connect_per_voi_y
        self.compute_distribution_params()


class Connectivity:

    # kind: {'covariance', 'correlation', 'partial correlation', 'precision'}
    # estimator: None or sklearn.covariance estimator object
    # corr_method: {'pearson', 'spearman'}

    def __init__(self,
                 feature_df: pd.DataFrame,
                 covariate_df=None,
                 kind='correlation',
                 estimator=None,
                 corr_method='pearson',
                 fisher_transf=False,
                 n_bootstrap_samples=None,
                 random_state=0,
                 n_min_unique_elements=3,
                 lin_fit_method=None,
                 name=None,
                 ):

        self.feature_df = feature_df
        self.covariate_df = covariate_df

        self.kind = kind
        self.estimator = estimator
        self.corr_method = corr_method
        self.fisher_transf = fisher_transf
        self.name = name

        # BOOTSTRAPPING ATTRIBUTES
        self.n_boots = n_bootstrap_samples
        self.seed = random_state
        self.n_min_unique_elements = n_min_unique_elements

        self.subjects = self.feature_df.index.to_list()
        self.vois = self.feature_df.columns.to_list()
        self.n_subjects = len(self.subjects)
        self.n_vois = len(self.vois)
        self.matrix, self.bootstrap_matrices = self.calculate_connectivity()

        if lin_fit_method is not None:
            self.calculate_linear_fit(method=lin_fit_method)

        self.df = pd.DataFrame(self.matrix, index=self.vois, columns=self.vois)
        self.network_nonthr = Network(self.df, name='Full Network Non-Thresholded')

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return 'Unnamed'

    # TODO: Covariates

    @np.errstate(invalid='raise', divide='raise')
    def calculate_connectivity_single_bootstrap_sample(self, feature_df, covariate_df):
        if self.estimator is None:

            matrix = get_correlation_matrix(data=feature_df,
                                            cov_df=covariate_df,
                                            kind=self.kind,
                                            method=self.corr_method)

        else:
            if covariate_df is not None:
                warnings.warn('Warning: This type of correlation matrix calculation is not implemented with covariates. The covariates you provided were ignored')
            scaler = StandardScaler()
            boot_features_df = scaler.fit_transform(feature_df)
            connectivity_measure = ConnectivityMeasure(cov_estimator=self.estimator,
                                                       kind=self.kind,
                                                       # standardize='zscore_sample',
                                                       )
            matrix = connectivity_measure.fit_transform([boot_features_df])[0]
        np.fill_diagonal(matrix, 0)  # setting autocorrelations to zero
        if self.fisher_transf:
            return(np.arctanh(matrix))
        else:
            return matrix

    def calculate_linear_fit_single_bootstrap_sample(self, feature_df):
        feature_arr = feature_df.to_numpy()
        n_vois = self.n_vois
        K = np.zeros((n_vois, n_vois))
        B = np.zeros((n_vois, n_vois))
        for i in range(n_vois):
            for j in range(i + 1, n_vois):
                feature_i = feature_arr[:, i]
                feature_j = feature_arr[:, j]
                K[j, i], B[j, i] = linreg(feature_i, feature_j)
        K = K + K.T
        B = B + B.T
        return K, B

    def calculate_connectivity(self):
        if self.n_boots == 1 or self.n_boots is None:
            matrix = self.calculate_connectivity_single_bootstrap_sample(self.feature_df, covariate_df=self.covariate_df)
            matrices = np.array([matrix])
            return matrix, matrices
        if not isinstance(self.seed, list):
            np.random.seed(self.seed)  # compatibility addition
        matrices = []
        for i in range(self.n_boots):
            if isinstance(self.seed, list):
                seed = self.seed[i]  # compatibility addition
            else:
                seed = None  # compatibility addition
            boot_sample = self.draw_bootstrap_sample(self.n_subjects, self.n_min_unique_elements, random_seed=seed)
            #print(seed)
            #print(boot_sample)
            boot_feature_df = self.feature_df.iloc[boot_sample]
            if self.covariate_df is None:
                boot_covariate_df = None
            else:
                boot_covariate_df = self.covariate_df.iloc[boot_sample]
            matrix = self.calculate_connectivity_single_bootstrap_sample(boot_feature_df, covariate_df=boot_covariate_df)
            matrices.append(matrix)
        matrices = np.array(matrices, dtype=np.float32)
        #if self.fisher_transf:
        #    matrices = check_for_infs_in_matrix(matrices)
        mean_matrix = np.mean(matrices, axis=0)
        #mean_matrix = np.mean(np.ma.masked_invalid(np.array(matrices)), axis=0)
        return mean_matrix, matrices


    def calculate_linear_fit(self, method='bootstrapping'):
        # possible methods:
        # 'bootstrapping', 'LinearRegression', 'LassoCV', 'RidgeCV'

        if method == 'bootstrapping':
            if self.n_boots == 1 or self.n_boots is None:
                K, B = self.calculate_linear_fit_single_bootstrap_sample(self.feature_df)
                return K, B
            if not isinstance(self.seed, list):
                np.random.seed(self.seed)
            Ks = []
            Bs = []
            for i in range(self.n_boots):
                if isinstance(self.seed, list):
                    seed = self.seed[i]  # compatibility addition
                else:
                    seed = None  # compatibility addition
                boot_sample = self.draw_bootstrap_sample(self.n_subjects, self.n_min_unique_elements, random_seed=seed)
                #print(seed)
                #print(boot_sample)
                boot_feature_df = self.feature_df.iloc[boot_sample]
                K, B = self.calculate_linear_fit_single_bootstrap_sample(boot_feature_df)
                Ks.append(K)
                Bs.append(B)
            Ks = np.array(Ks, dtype=np.float32)
            Bs = np.array(Bs, dtype=np.float32)
            self.K = np.mean(Ks, axis=0)
            self.B = np.mean(Bs, axis=0)

        else:
            try:
                Model = getattr(sklearn.linear_model, method)
            except AttributeError:
                raise Exception(f'Unknown method {method} for linear fitting')
            feature_arr = self.feature_df.to_numpy()
            n_vois = self.n_vois
            K = np.zeros((n_vois, n_vois))
            B = np.zeros((n_vois, n_vois))
            for i in range(n_vois):
                for j in range(i + 1, n_vois):
                    feature_i = feature_arr[:, i]
                    feature_j = feature_arr[:, j]
                    #model = Model(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(feature_i.reshape(-1, 1), feature_j)
                    model = Model().fit(feature_i.reshape(-1, 1), feature_j)
                    K[j, i] = np.squeeze(model.coef_)
                    B[j, i] = np.squeeze(model.intercept_)
                    #print(model.alpha_)
            self.K = K + K.T
            self.B = B + B.T

        self.df_K = pd.DataFrame(self.K, index=self.vois, columns=self.vois)
        self.df_B = pd.DataFrame(self.B, index=self.vois, columns=self.vois)
        self.lin_fit_method = method

    def threshold_connectivity(self, method='CI', p=0.005, matrix_thr=0.5, matrix_thr_type='both'):
        self.thr_method = method
        if method == 'CI':
            if not self.fisher_transf:
                raise Exception('Connectivity matrix must be Fisher-transformed before applying CI-thresholding')
            if self.n_boots == 1 or self.n_boots is None:
                raise Exception('CI-thresholding cannot be applied to a non-bootstrapped matrix')

            lower_percentile = 100 * p / 2
            upper_percentile = 100 * (1 - p / 2)
            matrix_ci = np.percentile(self.bootstrap_matrices, [lower_percentile, upper_percentile], axis=0)
            mask = np.logical_or(matrix_ci[0] > 0, matrix_ci[1] < 0)
            self.thr_p_value = p

        elif method == 'SICE':
            scaler = StandardScaler()
            standardized_feature_df = scaler.fit_transform(self.feature_df)
            connectivity_measure = ConnectivityMeasure(cov_estimator=GraphicalLassoCV(),
                                                       kind='precision',
                                                       # standardize='zscore_sample',
                                                       )
            precision_matrix = connectivity_measure.fit_transform([standardized_feature_df])[0]
            mask = np.abs(precision_matrix) > 0.001  # 0.001 instead of 0 due to finite precision

        elif method == 'matrix_threshold':  # analogous to r-threshold in old code
            if matrix_thr_type == 'both':
                mask = np.logical_or(self.matrix > matrix_thr, self.matrix < -matrix_thr)
            elif matrix_thr_type == 'lower':
                mask = self.matrix < -matrix_thr
            elif matrix_thr_type == 'upper':
                mask = self.matrix > matrix_thr
            else:
                raise Exception(f'Matrix threshold type {matrix_thr_type} is not supported. Valid options are "lower", "upper", "both".')
            self.thr_matrix_value = matrix_thr
            self.thr_matrix_type = f'{matrix_thr_type} threshold'

        else:
            raise Exception('Select a valid thresholding method. Valid options are "CI", "SICE", "matrix_threshold".')

        if not hasattr(self, 'df_thr'):
            self.df_thr = self.df.copy()

        self.df_thr *= mask
        self.network = Network(self.df_thr, name='Full Network')

        try:
            if not hasattr(self, 'df_K_thr'):
                self.df_K_thr = self.df_K.copy()
            if not hasattr(self, 'df_B_thr'):
                self.df_B_thr = self.df_B.copy()
            self.df_K_thr *= mask
            self.df_B_thr *= mask
            self.network.df_K = self.df_K_thr
            self.network.df_B = self.df_B_thr
        except AttributeError:
            pass
        #self.matrix_thresholded_binary = self.matrix_thresholded != 0
        #self.n_connections = np.sum(np.tril(self.matrix_thresholded_binary))
        #self.n_voi_connections = dict(zip(self.vois, np.sum(self.matrix_thresholded_binary, axis=0)))

    def evaluate_subnetworks(self, subnet_dict: dict):
        self.subnetworks = {}
        self.subnetworks_nonthr = {}
        self.n_connect_subnetworks = {}
        for subnet_1_name, subnet_1_vois in subnet_dict.items():
            for subnet_2_name, subnet_2_vois in subnet_dict.items():
                if hasattr(self, 'thr_method'):
                    subnetwork = init_network(self.df_thr.loc[subnet_1_vois, subnet_2_vois], subnet_1_name, subnet_2_name)
                    if hasattr(self, 'lin_fit_method'):
                        subnetwork.df_K = self.df_K_thr.loc[subnet_1_vois, subnet_2_vois]
                        subnetwork.df_B = self.df_B_thr.loc[subnet_1_vois, subnet_2_vois]
                    self.subnetworks[subnetwork.name] = subnetwork
                    self.n_connect_subnetworks[subnetwork.name] = subnetwork.n_connect
                subnetwork_nonthr = init_network(self.df.loc[subnet_1_vois, subnet_2_vois], subnet_1_name, subnet_2_name)
                if hasattr(self, 'lin_fit_method'):
                    subnetwork_nonthr.df_K = self.df_K.loc[subnet_1_vois, subnet_2_vois]
                    subnetwork_nonthr.df_B = self.df_B.loc[subnet_1_vois, subnet_2_vois]
                self.subnetworks_nonthr[subnetwork_nonthr.name] = subnetwork_nonthr

    @staticmethod
    def draw_bootstrap_sample(n_subjects, n_min_unique_elements=3, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)  # compatibility addition
        bootstrap_sample = np.random.choice(n_subjects, size=n_subjects, replace=True)
        n_unique_elements = 0
        while n_unique_elements < n_min_unique_elements:
            n_unique_elements = len(np.unique(bootstrap_sample))
            if n_unique_elements < n_min_unique_elements:
                random_seed += 1
                np.random.seed(random_seed)  # compatibility addition
                bootstrap_sample = np.random.choice(n_subjects, size=n_subjects, replace=True)
            else:
                #print(random_seed)
                #print(bootstrap_sample)
                return bootstrap_sample


