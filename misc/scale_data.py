from cohort import Cohort
from atlas import Atlas
from pathlib import Path
from glob import glob
import pandas as pd


if __name__ == '__main__':

    root_dir = Path('/Users/artem/Project_Data/Microglia_Synchronicity')
    data_dir = root_dir / 'data' / 'Mouse'
    scaled_output_dir = data_dir / 'GLM_scaled'
    aver_output_dir = scaled_output_dir / 'Cohort_Average'


    scaling_atlas_dir = Path('/Users/artem/PycharmProjects/molecular_connectivity/data/atlas/Global_Mean.nii')
    scaling_atlas = Atlas(scaling_atlas_dir)


    master_xlsx_path = f'{root_dir}/20240516_mouse_cohorts_summary.xlsx'
    master_xlsx = pd.ExcelFile(master_xlsx_path)

    study_names = master_xlsx.sheet_names

    study_dfs = {}
    studies = []
    for study_name in study_names:
        study_dfs[study_name] = pd.read_excel(master_xlsx_path, sheet_name=study_name)
        study_dict = {}
        cohort_names = study_dfs[study_name].columns.to_list()
        for cohort_name in cohort_names:
            study_dict[cohort_name] = study_dfs[study_name][cohort_name].to_list()
        studies.append(study_dict)


    all_paths = []
    for study_name, study_df in study_dfs.items():
        # if study_name not in ['Moderate']:
        #      continue
        print(f'\n\nSTUDY: {study_name}')
        filenames = glob(f'{data_dir}/*.nii')
        study_cohorts = []

        # for cohort_name, cohort_nim_stems in study.items():
        for i, cohort_name in enumerate(study_df.columns):
            cohort_nim_stems = study_df[cohort_name].to_list()
            # cohort_nim_paths = [nim_path for nim_path in filenames if cohort_name in nim_path]
            cohort_nim_paths = [f'{data_dir}/{cohort_nim_stem}.nii' for cohort_nim_stem in cohort_nim_stems]
            cohort = Cohort(cohort_nim_paths, cohort_name=cohort_name)
            cohort.load_image_arrays()
            cohort.scale(atlas=scaling_atlas, feature='mean')
            cohort.generate_average_image(save_path=aver_output_dir)
            all_paths += cohort_nim_paths


    # cohort = Cohort(all_paths, cohort_name='Full data')
    # cohort.load_image_arrays()
    # cohort.scale(atlas=scaling_atlas, feature='mean', save_path=scaled_output_dir, verbose=True)
    #
    #
