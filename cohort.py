import warnings
from pathlib import Path
import os
import shutil
from pyexpat import features

import numpy as np
import pandas as pd
import random
from scipy import stats

import nibabel as nib

from connectivity import Connectivity
import functions as f


class Cohort:

    # features_name_col (string): column that includes img names (column values must be string that are contained in self.names.
    # E.g. if self.names=['sub-0001', 'sub-0002'] your identifier column might be df['features_name_col']=['0001', '0002'])
    def __init__(self,
                 nifti_image_list=None,
                 features_path=None,  # excel path
                 sheet_name=0,
                 extracted_features_path=None,
                 extracted_features_sheet_name=0,
                 features_name_col='name',
                 features_name_col_type=str,
                 features_not_found_behavior='exception',
                 use_filenames_as_ids=False,
                 cohort_name=None, ):

        if nifti_image_list is not None:
            self.nim_paths = [Path(nim_path) for nim_path in nifti_image_list]
            self.nims = [nib.squeeze_image(nib.load(path)) for path in self.nim_paths]
            self.names = []
            self.file_stems = []
            for nim_path in self.nim_paths:
                if nim_path.suffix == '.gz':
                    nim_path = nim_path.with_suffix('')
                self.file_stems.append(nim_path.stem)
            #self.names = [path.stem for path in self.nim_paths]
        elif extracted_features_path is not None:
            self.extracted_features = pd.read_excel(extracted_features_path, sheet_name=extracted_features_sheet_name, index_col=0)
            self.file_stems = self.extracted_features.index.to_list()
            self.file_stems = [str(stem) for stem in self.file_stems]
            self.extracted_features.index = self.file_stems
            self.nim_paths = None
            self.nims = None
            self.names = self.file_stems.copy()
            self.combined_features = self.extracted_features.copy()

        self.given_features = pd.DataFrame(index=self.names)
        if cohort_name is None:
            self.name = self.nim_paths[0].parent.name
        else:
            self.name = cohort_name
        if features_path is None:
            self.names = self.file_stems.copy()
        else:
            self.features_path = features_path
            self.features_name_col = features_name_col
            columns_dtypes = {self.features_name_col: features_name_col_type}
            df = pd.read_excel(features_path, sheet_name=sheet_name, dtype=columns_dtypes).set_index(features_name_col)
            valid_inds = []
            for name in self.file_stems:
                occurence_count = 0
                for name_in_df in df.index:
                    if str(name_in_df) in name:
                        occurence_count += 1
                        if occurence_count > 1:
                            raise Exception(f'Name {name} was found in column {features_name_col} more than one time. For each image, please ensure a unique row in your input table')
                        valid_inds.append(name_in_df)
                if occurence_count == 0:
                    if features_not_found_behavior == 'exception':
                        raise Exception(f'Name {name} was not found in column {features_name_col}. For each image, please ensure a unique row in your input table')
                    else:
                        warnings.warn(f'Name {name} was not found in column {features_name_col}.')
            self.names = valid_inds
            self.given_features = df.loc[valid_inds]
            if use_filenames_as_ids:
                self.given_features.set_index(pd.Index(self.file_stems), inplace=True)
        if not hasattr(self, 'extracted_features'):
            self.combined_features = self.given_features.copy()
        else:
            self.combined_features = pd.merge(self.given_features, self.extracted_features,
                                              how='outer', left_index=True, right_index=True, sort=False)
            self.combined_features = self.combined_features.reindex(self.given_features.index)
        self.n = len(self)

    def __len__(self):
        return len(self.names)
    def __str__(self):
        return f'Cohort {self.name}, n={len(self)}'
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return self.__dict__ == other.__dict__

    @classmethod
    def multiple_from_random_split(cls, nifti_image_list, n_subjects_per_split, features_path=None, features_name_col=None, seed=0, allow_overlap=False, cohort_name=None):
        if len(nifti_image_list) < n_subjects_per_split:
            return Cohort(nifti_image_list, features_path=features_path, features_name_col=features_name_col, cohort_name=cohort_name)
        elif allow_overlap:
            # EXPERIMENTAL: have to double-check
            nifti_image_set = set(nifti_image_list)
            if len(nifti_image_set) == n_subjects_per_split:
                n_subcohorts = 1
            else:
                n_subcohorts = len(nifti_image_set) // n_subjects_per_split + 1
            overlap = n_subjects_per_split - len(nifti_image_set) % n_subjects_per_split
            subcohorts = []
            random.seed(seed)
            for i in range(n_subcohorts):
                sample_set = set(random.sample(sorted(nifti_image_set), n_subjects_per_split))
                subcohorts.append(cls(sorted(sample_set), features_path=features_path, features_name_col=features_name_col, cohort_name=f'{cohort_name}_{i}'))
                set_to_remove = set(sorted(sample_set)[overlap:])
                nifti_image_set -= set_to_remove
            return subcohorts

        else:
            nifti_image_set = set(nifti_image_list)
            n_subcohorts = len(nifti_image_set) // n_subjects_per_split
            subcohorts = []
            random.seed(seed)
            for i in range(n_subcohorts):
                sample_set = set(random.sample(sorted(nifti_image_set), n_subjects_per_split))
                subcohorts.append(cls(sorted(sample_set), features_path=features_path, features_name_col=features_name_col, cohort_name=f'{cohort_name}_{i}'))
                nifti_image_set -= sample_set
            return subcohorts

    def load_image_arrays(self):
        self.nim_arrs = [nim.get_fdata() for nim in self.nims]

    def register_to_template(self, template_nii, save_path=None, inplace=True, **kwargs):
        import ants
        if isinstance(template_nii, nib.Nifti1Image):
            templ = ants.from_nibabel(template_nii)
        elif isinstance(template_nii, ants.ANTsImage):
            templ = template_nii
        elif isinstance(template_nii, Path) or isinstance(template_nii, str):
            templ = ants.image_read(template_nii)
        else:
            raise TypeError('ants.ANTsImage, nib.Nifti1Image, or Path was expected')

        im_regs = []
        self.fwdtransfs = []
        self.invtransfs = []
        for i in range(self.n):
            nim = self.nims[i]
            name = self.names[i]
            im = ants.from_nibabel(nim)
            print(f'Registering image {name}, {i+1} out of {self.n}')
            reg_output = ants.registration(fixed=templ, moving=im, **kwargs)
            im_reg = reg_output['warpedmovout']
            fwdtransf_paths = reg_output['fwdtransforms']
            invtransf_paths = reg_output['invtransforms']
            # WRITING IMAGE
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if 'type_of_transform' in kwargs:
                    reg_suffix = kwargs.get('type_of_transform')
                else:
                    reg_suffix = 'reg'
                ants.image_write(im_reg, f'{save_path}/{reg_suffix}_{name}.nii')
                # WRITING TRANSFORMS
                for fwdtransf_path, invtransf_path in zip(fwdtransf_paths, invtransf_paths):
                    # fwd1 must be "i", not "1"?
                    shutil.copy(fwdtransf_path, f'{save_path}/{reg_suffix}_{name}_fwd1_{Path(fwdtransf_path).name}')
                    shutil.copy(invtransf_path, f'{save_path}/{reg_suffix}_{name}_inv1_{Path(invtransf_path).name}')
            self.fwdtransfs.append(fwdtransf_paths)
            self.invtransfs.append(invtransf_paths)
            if inplace:
                self.nims[i] = ants.to_nibabel(im_reg)
            else:
                im_regs.append(im_reg)
        if not inplace:
            return im_regs

        # FreeSurfer?

    def reorient_to_atlas_x(self, atlas, save_path=None, **kwargs):

        for i, nim in enumerate(self.nims):
            self.nims[i] = f.reorient_to_target(input_nifti=nim, target_nifti=atlas.nim, **kwargs)
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                nib.save(self.nims[i], (Path(save_path) / self.names[i]).with_suffix('.nii.gz'))

    def reorient_to_atlas(self, atlas, verbose=False, save_path=None, set_atlas_affine=False):
        target_ornt = nib.orientations.io_orientation(atlas.nim.affine)
        if verbose: print(f'Target orientation: {nib.orientations.aff2axcodes(atlas.nim.affine)}')
        for i, nim in enumerate(self.nims):
            input_ornt = nib.orientations.io_orientation(nim.affine)
            transform = nib.orientations.ornt_transform(input_ornt, target_ornt)
            if verbose: print(f'Image {self.names[i]} with orientation {nib.orientations.aff2axcodes(nim.affine)} is reoriented')
            self.nims[i] = nim.as_reoriented(transform)
            if set_atlas_affine:
                self.nims[i] = nib.Nifti1Image(dataobj=nim.dataobj, affine=atlas.nim.affine, header=nim.header)
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                nib.save(self.nims[i], (Path(save_path) / self.names[i]).with_suffix('.nii.gz'))

    def scale(self, atlas=None, label=1, feature='mean', inj_dose_col='', weight_col='', inj_dose_unit='MBq', weight_unit='kg', save_path=None, verbose=False):
        # percent_ID scaling
        if inj_dose_col:
            self.scaling_factors = np.array(self.given_features.loc[self.names, inj_dose_col])
            if inj_dose_unit == 'MBq':
                self.scaling_factors *= 1000
            elif inj_dose_unit == 'kBq':
                pass
            else:
                raise KeyError('Unrecognized unit for injected dose')
            print(f'Injected dose in {inj_dose_unit}')
            self.scaling_feature_name = 'percent_ID'

            # SUV scaling
            if weight_col:
                weights = np.array(self.given_features.loc[self.names, weight_col])
                if weight_unit == 'kg':
                    weights *= 1000
                elif weight_unit == 'g':
                    pass
                else:
                    raise KeyError('Unrecognized unit for weight')
                print(f'Subject weight in {weight_unit}')
                self.scaling_factors /= weights
                self.scaling_feature_name = 'SUV'
            else:
                self.scaling_factors /= 100  # to get percentage of ID

        # VOI scaling (SUV ratio)
        else:
            self.scaling_factors = np.array([f.extract(arr, atlas, label=label, feature=feature) for arr in self.nim_arrs])
            self.scaling_feature_name = f'{atlas.name}_region_{label}_{feature}'

        # perform the scaling
        for i in range(self.n):
            if verbose:
                print(f'Scaling image {i} ({self.names[i]}) of {self.n}')
            self.nim_arrs[i] /= self.scaling_factors[i]
            self.nims[i] = nib.Nifti1Image(self.nim_arrs[i], self.nims[i].affine, self.nims[i].header)  # think about keeping DICOM header
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                nib.save(self.nims[i], (Path(save_path) / self.names[i]).with_suffix('.nii.gz'))

        self.scaling_factors = pd.Series(self.scaling_factors, index=self.names)
        self.given_features['scaling_factor'] = self.scaling_factors
        self.combined_features['scaling_factor'] = self.scaling_factors
        # self.combined_features = pd.merge(self.combined_features, self.given_features, how='outer', left_index=True, right_index=True)


    def generate_average_image(self, save_dir=None, add_n_subjects=False):
        aver_arr = np.mean(np.array(self.nim_arrs), axis=0)
        aver_nim = nib.Nifti1Image(dataobj=aver_arr, affine=self.nims[0].affine, header=self.nims[0].header)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if add_n_subjects:
                n_sbj = f'_n={self.n}'
            else:
                n_sbj = ''
            nib.save(aver_nim, (Path(save_dir) / f'average_{self.name}{n_sbj}').with_suffix('.nii.gz'))


    def extract_features(self, atlas, feature='mean', labels='all', inplace=True, cols_to_drop=None, save_path=None, prefix=None, **extraction_kwargs):
        # provide list of labels (=VOIs) if only a subset of VOIs is needed
        # try with own method first (e.g. "mean" with numpy), otherwise fallback to pyradiomics
        for nim, nim_name in zip(self.nims, self.names):
            assert (nim.affine == atlas.nim.affine).all(), (f'Affine matrix of image {nim_name} does not equal affine matrix of atlas.'
                                                            f'\n\nImage affine\n{nim.affine}\n\nAtlas affine\n{atlas.nim.affine}')
        if labels == 'all':
            labels = atlas.vois
        extracted_features = []
        for i, arr in enumerate(self.nim_arrs):
            extracted_features_per_subject = [f.extract(arr, atlas, label=voi, feature=feature, **extraction_kwargs) for voi in labels]
            extracted_features.append(np.array(extracted_features_per_subject))
        extracted_features = pd.DataFrame(data=np.array(extracted_features), index=self.names, columns=labels)
        if cols_to_drop is not None:
            extracted_features.drop(columns=cols_to_drop, inplace=True)

        if prefix is None:
            prefix = ''
        else:
            prefix = f'{prefix}_'

        if inplace:
            self.extracted_features = extracted_features
            self.extracted_feature_name = feature
        else:
            return extracted_features.add_prefix(prefix)
        #voi_col_names = {old:f'VOI_{new}' for old, new in zip(extracted_features.columns, extracted_features.columns)}
        #self.combined_features = pd.merge(self.given_features, extracted_features.rename(columns=voi_col_names), left_index=True, right_index=True)

        if inplace:
            inds = self.combined_features.index if not self.combined_features.empty else self.extracted_features.index
            self.combined_features = pd.merge(self.combined_features, extracted_features.add_prefix(prefix), how='outer',
                                              left_index=True, right_index=True, sort=False).reindex(inds)
            self.combined_features_long = pd.melt(self.combined_features, id_vars=self.given_features.columns,
                                                  value_vars=extracted_features.add_prefix(prefix).columns, var_name='VOI',
                                                  value_name=feature, ignore_index=False)
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))
            extracted_features.T.to_excel(save_path)

    def calculate_connectivity(self,
                               covariates=None,  # list
                               **kwargs):
        self.covariates = covariates
        if covariates is None:
            cov_df = None
        else:
            cov_df = self.given_features[covariates].copy()
        self.connectivity = Connectivity(feature_df=self.extracted_features, covariate_df=cov_df, **kwargs)
        self._connectivity_init_kwargs = kwargs


    def calculate_average_cds_matrix(self, ref_cohort, loo_verbose=False, excel_out_path=None):


        if self == ref_cohort:
            if loo_verbose:
                print('Verbose mode is selected. Note that thresholding parameters are hard-coded')
            # leave-one-out
            self._loo_conns_test = {}
            Ds = {}
            for subject in self.names:
                if loo_verbose:
                    print(f'Calculating CDS using LOO approach for subject {subject}')
                extracted_features_loo = self.extracted_features.copy().drop(subject, axis=0)
                extracted_features_sbj = self.extracted_features.loc[[subject]].copy()
                if self.covariates is None:
                    cov_df_loo = None
                else:
                    cov_df_loo = self.given_features[self.covariates].copy().drop(subject, axis=0)

                conn_loo = Connectivity(feature_df=extracted_features_loo,
                                        covariate_df=cov_df_loo,
                                        **self._connectivity_init_kwargs)
                conn_loo.calculate_linear_fit()

                Ds.update(f.distance_matrix(features=extracted_features_sbj,
                                       K=conn_loo.df_K,
                                       B=conn_loo.df_B,
                                       ))


                if loo_verbose:
                    conn_loo.threshold_connectivity(method='CI', p=0.005)
                    conn_loo.threshold_connectivity(method='matrix_threshold', matrix_thr=0.5)
                    print(f'total: {conn_loo.network.n_connect}')

                self._loo_conns_test[subject] = conn_loo
            Ds_arr = np.array([D.to_numpy() for D in Ds.values()])
            self.sum_cds_arr = np.nansum(Ds_arr, axis=0)


        else:
            self.sum_cds_arr = f.sum_cds_matrix_per_cohort(features=self.extracted_features,
                                     K=ref_cohort.connectivity.df_K,
                                     B=ref_cohort.connectivity.df_B, )

        self.aver_cds_arr = self.sum_cds_arr / self.n
        self.aver_cds = pd.DataFrame(data=self.aver_cds_arr,
                                     index=self.extracted_features.columns,
                                     columns=self.extracted_features.columns,
                                     )
        if excel_out_path is not None:
            if not os.path.exists(os.path.dirname(excel_out_path)):
                os.mkdir(os.path.dirname(excel_out_path))
            self.aver_cds.to_excel(excel_out_path)


    def calculate_cds(self, ref_cohort, subnetwork_name=None, prefix=None, loo_verbose=False, calculate_pairwise_cds=False):
        # CDS = Connectivity Deviation Score
        # reference cohort needs to be a cohort object

        if self == ref_cohort:
            if loo_verbose:
                print('Verbose mode is selected. Note that thresholding parameters are hard-coded')
            # leave-one-out
            self.cds = pd.DataFrame()
            self._loo_conns_test = {}
            if calculate_pairwise_cds:
                Ds = {}
            for subject in self.names:
                if loo_verbose:
                    print(f'Calculating CDS using LOO approach for subject {subject}')
                extracted_features_loo = self.extracted_features.copy().drop(subject, axis=0)
                extracted_features_sbj = self.extracted_features.loc[[subject]].copy()
                if self.covariates is None:
                    cov_df_loo = None
                else:
                    cov_df_loo = self.given_features[self.covariates].copy().drop(subject, axis=0)

                conn_loo = Connectivity(feature_df=extracted_features_loo,
                                        covariate_df=cov_df_loo,
                                        **self._connectivity_init_kwargs)
                conn_loo.calculate_linear_fit()
                if subnetwork_name is None:
                    cds_sbj = f.cds_per_voi(features=extracted_features_sbj,
                                            K=conn_loo.df_K,
                                            B=conn_loo.df_B,)
                    if calculate_pairwise_cds:
                        D = f.distance_matrix(features=extracted_features_sbj,
                                              K=conn_loo.df_K,
                                              B=conn_loo.df_B,)
                else:
                    try:
                        #subnet_dict = {name: network.vois for name, network in
                        #               self.connectivity.subnetworks_nonthr.items()}
                        subnet_dict = {subnetwork_name: self.connectivity.subnetworks_nonthr[subnetwork_name].vois}
                        if loo_verbose:
                            conn_loo.threshold_connectivity(method='CI', p=0.005)
                            conn_loo.threshold_connectivity(method='matrix_threshold', matrix_thr=0.5)
                            print(f'CDS in subnetwork {subnetwork_name}')
                        conn_loo.evaluate_subnetworks(subnet_dict=subnet_dict)
                        if loo_verbose:
                            print(f'subnet: {conn_loo.subnetworks[subnetwork_name].n_connect}')
                        cds_sbj = f.cds_per_voi(features=extracted_features_sbj,
                                                 K=conn_loo.subnetworks_nonthr[subnetwork_name].df_K,
                                                 B=conn_loo.subnetworks_nonthr[subnetwork_name].df_B, )
                        if calculate_pairwise_cds:
                            D = f.distance_matrix(features=extracted_features_sbj,
                                                     K=conn_loo.subnetworks_nonthr[subnetwork_name].df_K,
                                                     B=conn_loo.subnetworks_nonthr[subnetwork_name].df_B, )
                    except KeyError:
                        raise KeyError(f'Selected network {subnetwork_name} is not defined')

                if loo_verbose:
                    conn_loo.threshold_connectivity(method='CI', p=0.005)
                    conn_loo.threshold_connectivity(method='matrix_threshold', matrix_thr=0.5)
                    print(f'total: {conn_loo.network.n_connect}')
                self._loo_conns_test[subject] = conn_loo
                self.cds = pd.concat([self.cds, cds_sbj], ignore_index=False)
                if calculate_pairwise_cds:
                    Ds.update(D)
            self.cds_pairwise = f.pairwise_cds_distr_params(Ds=Ds)

        else:
            if subnetwork_name is None:
                self.cds = f.cds_per_voi(features=self.extracted_features,
                                         K=ref_cohort.connectivity.df_K,
                                         B=ref_cohort.connectivity.df_B,)
                if calculate_pairwise_cds:
                    self.cds_pairwise = f.pairwise_cds_distr_params(features=self.extracted_features,
                                                                    K=ref_cohort.connectivity.df_K,
                                                                    B=ref_cohort.connectivity.df_B, )

            else:
                try:
                    self.cds = f.cds_per_voi(features=self.extracted_features,
                                             K=ref_cohort.connectivity.subnetworks_nonthr[subnetwork_name].df_K,
                                             B=ref_cohort.connectivity.subnetworks_nonthr[subnetwork_name].df_B, )
                    self.cds_pairwise = f.pairwise_cds_distr_params(features=self.extracted_features,
                                             K=ref_cohort.connectivity.subnetworks_nonthr[subnetwork_name].df_K,
                                             B=ref_cohort.connectivity.subnetworks_nonthr[subnetwork_name].df_B, )

                except KeyError:
                    raise KeyError(f'Selected network {subnetwork_name} is not defined')
        self.cds_ref_cohort_name = ref_cohort.name

        if subnetwork_name is None:
            try:
                self.cds_network_name = ref_cohort.connectivity.network.name
            except AttributeError:
                self.cds_network_name = ref_cohort.connectivity.network_nonthr.name
        else:
            self.cds_network_name = subnetwork_name
        if prefix is None:
            value_name = 'CDS'
            prefix = f'cds_{self.cds_network_name}_'
        else:
            value_name = prefix
            prefix = f'{prefix}_'
        self.combined_features = pd.merge(self.combined_features, self.cds.add_prefix(prefix), how='outer',
                                          left_index=True, right_index=True, sort=False).reindex(self.combined_features.index)
        self.cds_long = pd.melt(self.cds, value_vars=self.cds.columns, var_name='VOI', value_name=value_name, ignore_index=False)
        #self.combined_features_long = pd.merge(self.combined_features_long, self.cds_long,
        #                                       left_index=True, right_index=True,
        #                                       on='VOI', how='outer')
        # TODO: implement "combined_features_long"

    def add_features(self, other=None, feature_names=None, feature_df=None, prefix_to_add=''):
        if feature_df is not None:
            df = feature_df.copy()
        else:
            if feature_names is not None and other is not None:
                df = other.combined_features[feature_names].copy()
            else:
                raise Exception('Either feature_names or feature_df must be defined')

        df = df.add_prefix(prefix_to_add)

        self.combined_features = self.combined_features.join(df, how='left')

    # Kullback-Leibler similarity
    def calculate_kls(self, atlas=None, long_df_path=None, feature_col=None, voi_col=None, subject_col=None, n_sample_points=1000):
        self.kls = {}
        if self.nim_paths is not None:
            for arr, name in zip(self.nim_arrs, self.names):
                adjacency_matrix = np.zeros((len(atlas), len(atlas)))
                sample_points = np.linspace(np.min(arr), np.max(arr), n_sample_points)
                for i, voi_1 in enumerate(atlas.vois):
                    inds_1 = atlas.mask[voi_1]
                    values_1 = arr[inds_1]
                    pdf_1 = stats.gaussian_kde(values_1)

                    for j, voi_2, in enumerate(atlas.vois):
                        inds_2 = atlas.mask[voi_2]
                        values_2 = arr[inds_2]
                        pdf_2 = stats.gaussian_kde(values_2)

                        kld = f.symmetric_kl(pdf_1, pdf_2, sample_points)
                        similarity = np.exp(-kld)

                        adjacency_matrix[i, j] = similarity
                df_adjacency = pd.DataFrame(data=adjacency_matrix, index=atlas.vois, columns=atlas.vois)
        elif long_df_path is not None:
            long_df = pd.read_csv(long_df_path, index_col=0)
            vois = long_df[voi_col].unique()
            n_vois = len(vois)

            for subject_name in self.names:

                df_sbj = long_df.loc[long_df[subject_col] == subject_name]

                adjacency_matrix = np.zeros((n_vois, n_vois))
                sample_points = np.linspace(df_sbj[feature_col].min(), df_sbj[feature_col].max(), n_sample_points)
                for i, voi_1 in enumerate(vois):
                    values_1 = df_sbj.loc[df_sbj[voi_col] == voi_1, feature_col].to_numpy()
                    pdf_1 = stats.gaussian_kde(values_1)

                    for j, voi_2, in enumerate(vois):
                        values_2 = df_sbj.loc[df_sbj[voi_col] == voi_2, feature_col].to_numpy()
                        pdf_2 = stats.gaussian_kde(values_2)

                        kld = f.symmetric_kl(pdf_1, pdf_2, sample_points)
                        similarity = np.exp(-kld)

                        adjacency_matrix[i, j] = similarity
                df_adjacency = pd.DataFrame(data=adjacency_matrix, index=vois, columns=vois)
                self.kls[subject_name] = df_adjacency




