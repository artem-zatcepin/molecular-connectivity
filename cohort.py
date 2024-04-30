import warnings
from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd
import random

import nibabel as nib

from connectivity import Connectivity
import functions as f


class Cohort:

    def __init__(self,
                 nifti_image_list,
                 features_path=None,  # excel path
                 features_name_col='name',  # column that includes img names (exactly like in self.names)
                 cohort_name=None,):

        self.nim_paths = [Path(nim_path) for nim_path in nifti_image_list]
        self.nims = [nib.squeeze_image(nib.load(path)) for path in self.nim_paths]
        self.names = []
        for nim_path in self.nim_paths:
            if nim_path.suffix == '.gz':
                nim_path = nim_path.with_suffix('')
            self.names.append(nim_path.stem)
        #self.names = [path.stem for path in self.nim_paths]
        if cohort_name is None:
            self.name = self.nim_paths[0].parent.name
        else:
            self.name = cohort_name
        self.n = len(self)
        if features_path is not None:
            df = pd.read_excel(features_path)
            self.given_features = df[df[features_name_col].isin(self.names)].set_index(features_name_col)
            assert len(self.given_features) == len(self.names), f'Not all subjects are in {os.path.basename(features_path)}'

    def __len__(self):
        return len(self.nim_paths)
    def __str__(self):
        return f'Cohort {self.name}, n={len(self)}'

    @classmethod
    def multiple_from_random_split(cls, nifti_image_list, n_subjects_per_split, seed=0, allow_overlap=False, cohort_name=None):
        if len(nifti_image_list) < n_subjects_per_split:
            return Cohort(nifti_image_list, cohort_name=cohort_name)
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
                subcohorts.append(cls(sorted(sample_set), cohort_name=f'{cohort_name}_{i}'))
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
                subcohorts.append(cls(sorted(sample_set), cohort_name=f'{cohort_name}_{i}'))
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

    def scale(self, atlas=None, label=1, feature='mean', inj_dose_col='', weight_col='', inj_dose_unit='MBq', weight_unit='kg', save_path=None):
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

        # VOI scaling (SUV ratio)
        else:
            self.scaling_factors = np.array([f.extract(arr, atlas, label=label, feature=feature) for arr in self.nim_arrs])
            self.scaling_feature_name = f'{atlas.name}_region_{label}_{feature}'
        if not hasattr(self, 'given_features'):
            self.given_features = pd.DataFrame()
        self.given_features['scaling_factor'] = self.scaling_factors

        # perform the scaling
        for i in range(self.n):
            self.nim_arrs[i] /= self.scaling_factors[i]
            self.nims[i] = nib.Nifti1Image(self.nim_arrs[i], self.nims[i].affine, self.nims[i].header)  # think about keeping DICOM header
            if save_path is not None:
                nib.save(self.nims[i], (Path(save_path) / self.names[i]).with_suffix('.nii.gz'))

    def extract_features(self, atlas, feature='mean', labels='all', inplace=True, save_path=None, **extraction_kwargs):
        # provide list of labels (=VOIs) if only a subset of VOIs is needed
        # try with own method first (e.g. "mean" with numpy), otherwise fallback to pyradiomics
        for nim in self.nims:
            assert (nim.affine == atlas.nim.affine).all()
        if labels == 'all':
            labels = atlas.vois
        extracted_features = []
        for i, arr in enumerate(self.nim_arrs):
            extracted_features_per_subject = [f.extract(arr, atlas, label=voi, feature=feature, **extraction_kwargs) for voi in labels]
            extracted_features.append(np.array(extracted_features_per_subject))
        extracted_features = pd.DataFrame(data=np.array(extracted_features), index=self.names, columns=labels)
        if inplace:
            self.extracted_features = extracted_features
            self.extracted_feature_name = feature
        else:
            return extracted_features
        #voi_col_names = {old:f'VOI_{new}' for old, new in zip(extracted_features.columns, extracted_features.columns)}
        #self.combined_features = pd.merge(self.given_features, extracted_features.rename(columns=voi_col_names), left_index=True, right_index=True)
        self.combined_features = pd.merge(self.given_features, extracted_features, left_index=True, right_index=True)
        self.combined_features_long = pd.melt(self.combined_features, id_vars=self.given_features.columns,
                                              value_vars=extracted_features.columns, var_name='VOI',
                                              value_name=feature, ignore_index=False)
        if save_path is not None:
            extracted_features.T.to_excel(save_path)

    def calculate_connectivity(self, **kwargs):
        self.connectivity = Connectivity(feature_df=self.extracted_features, **kwargs)

    def calculate_cds(self, ref_cohort, subnetwork_name=None):
        # CDS = Connectivity Deviation Score
        # reference cohort needs to be a cohort object

        if subnetwork_name is None:
            self.cds = f.cds_per_voi(features=self.extracted_features,
                                     K=ref_cohort.connectivity.df_K,
                                     B=ref_cohort.connectivity.df_B,)
            self.cds_network_name = ref_cohort.connectivity.network.name
        else:
            try:
                self.cds = f.cds_per_voi(features=self.extracted_features,
                                         K=ref_cohort.connectivity.subnetworks_nonthr[subnetwork_name].df_K,
                                         B=ref_cohort.connectivity.subnetworks_nonthr[subnetwork_name].df_B, )
                self.cds_network_name = subnetwork_name
            except KeyError:
                raise KeyError(f'Selected network {subnetwork_name} is not defined')
        self.cds_ref_cohort_name = ref_cohort.name





