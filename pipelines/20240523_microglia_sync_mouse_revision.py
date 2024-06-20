from atlas import Atlas, AggregatedAtlas
from cohort import Cohort
import visualization as vis
import matplotlib.pyplot as plt
import seaborn as sns

import os
from glob import glob
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import nibabel as nib


if __name__ == '__main__':

    today = date.today().strftime("%Y%m%d")
    atlas_dir = Path('/Users/artem/PycharmProjects/molecular_connectivity/data/atlas/full_PVC_combined')
    atlases = []
    # scaling_voi_dict = dict(Onset = 'BS_PVC',
    #                         Moderate = 'R_SSE',
    #                         WT_PLX = 'CB_PVC',
    #                         TREM2 = 'L_THA',
    #                         APPPS1_TREM2 = 'R_AUD',
    #                         deltaE9_PLX = 'BFS',
    #                         PS2APP_PLX = 'BFS',
    #                         )
    scaling_voi_dict = None

    for i in range(1, 5):
        atlases.append(Atlas(f'{atlas_dir}/full_PVC_part{i}'))
    atlas = AggregatedAtlas(atlases,
                            legend_excel_path=f'{atlas_dir}/mouse_mirrione_pvc.xlsx',
                            aggregated_atlas_name='MaBenvenisteMirrioneModified',
                            )

    scaling_atlas_dir = Path('/Users/artem/PycharmProjects/molecular_connectivity/data/atlas/Global_Mean.nii')
    scaling_atlas = Atlas(scaling_atlas_dir)

    template_3d_path = Path('/Users/artem/PycharmProjects/molecular_connectivity/data/templates_for_3d/PMOD_T2_template_in_Matthias_space.nii')
    template_3d = nib.load(template_3d_path)

    root_dir = Path('/Users/artem/Project_Data/Microglia_Synchronicity')
    data_dir = root_dir / 'data' / 'Mouse'
    #features_path = root_dir / 'microglia_sync_human_info_anon.xlsx'
    #features_name_col = 'Anon_code'
    out_dir = root_dir / 'output' / f'{today}_mouse'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    i = 0
    cohorts = []

    n_boots = 10000
    np.random.seed(5)
    #randseeds = np.random.randint(0, 100000, size=100000)[:n_boots]
    randseeds = np.random.randint(0, 100000, size=100000)

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

    #%%
    #for study in studies[2:5]:
    anova_dict = {}
    ttest_dict = {}
    anova_signif_dict = {}
    ttest_signif_dict = {}

    network_for_di = 'AD_signature'

    for study_name, study_df in study_dfs.items():
        # if study_name not in ['Moderate']:
        #      continue
        print(f'\n\nSTUDY: {study_name}')
        filenames = glob(f'{data_dir}/*.nii')
        study_cohorts = []
        if scaling_voi_dict is not None:
            scaling_voi_name = scaling_voi_dict[study_name]
        else:
            scaling_voi_name = None

        #for cohort_name, cohort_nim_stems in study.items():
        for i, cohort_name in enumerate(study_df.columns):
            cohort_nim_stems = study_df[cohort_name].to_list()
            #cohort_nim_paths = [nim_path for nim_path in filenames if cohort_name in nim_path]
            cohort_nim_paths = [f'{data_dir}/{cohort_nim_stem}.nii' for cohort_nim_stem in cohort_nim_stems]
            cohort = Cohort(cohort_nim_paths, cohort_name=cohort_name)
            cohort.load_image_arrays()
            if scaling_voi_name is None:
                cohort.scale(atlas=scaling_atlas, feature='mean')
                cohort.extract_features(atlas=atlas, feature='mean',
                                        save_path=f'{out_dir}/means_mirrione_cohort_{cohort_name}.xlsx')
            else:
                cohort.scale(atlas=atlas, label=scaling_voi_name, feature='mean')
                cohort.extract_features(atlas=atlas, feature='mean',
                                        cols_to_drop=[scaling_voi_name],
                                        save_path=f'{out_dir}/means_mirrione_cohort_{cohort_name}.xlsx')
            print(f'Calculating connectivity for {cohort_name}')
            cohort.calculate_connectivity(kind='correlation',
                                          fisher_transf=True,
                                          n_bootstrap_samples=n_boots,
                                          #random_state=26,
                                          random_state=list(randseeds),
                                          name=cohort.name,)
            if i == 0:
                cohort.connectivity.calculate_linear_fit()
            cohort.connectivity.threshold_connectivity(method='CI', p=0.005)
            cohort.connectivity.threshold_connectivity(method='matrix_threshold', matrix_thr=0.5)
            cohort.connectivity.evaluate_subnetworks({network_for_di: atlas.networks[network_for_di]})
            print(cohort.connectivity.network.n_connect)
            print(f'In {network_for_di} network: {cohort.connectivity.subnetworks[network_for_di].n_connect}')
            study_cohorts.append(cohort)
            cohort.calculate_cds(ref_cohort=study_cohorts[0], subnetwork_name=network_for_di,
                                 prefix='DI', loo_verbose=True)

        cohorts += study_cohorts

    best_ref_vois = {}
    best_ref_vois_cohen = {}
    for study_name, ref_cohort_name in zip(study_dfs.keys(), ['WT_02M', 'WT_06M', 'Placebo', 'WT_12M', 'APPPS1_tr_TREM2_wt', 'FU1_TG_Plac', 'FU_TG_Plac']):
    #for study_name, ref_cohort_name in zip(['Onset'], ['WT_02M']):
    #for study_name, ref_cohort_name in zip(['Moderate'], ['WT_06M']):
        for cohort_name in study_dfs[study_name].columns:
        #for cohort_name in ['APPki_02_5M']:
        #for cohort_name in ['APPki_05M']:
            if cohort_name == ref_cohort_name:
                continue
            cohort = [cohort_tmp for cohort_tmp in cohorts if cohort_tmp.name == cohort_name][0]
            ref_cohort = [cohort_tmp for cohort_tmp in cohorts if cohort_tmp.name == ref_cohort_name][0]

            for coh in [cohort, ref_cohort]:
                df_tmp = coh.combined_features_long
                df_tmp['Cohort'] = [coh.name] * len(df_tmp)

                df_tmp = coh.cds_long
                df_tmp['Cohort'] = [coh.name] * len(df_tmp)

            df_suvr = pd.concat([ref_cohort.combined_features_long, cohort.combined_features_long], ignore_index=False)
            df_cds = pd.concat([ref_cohort.cds_long, cohort.cds_long], ignore_index=False)

            #for data, lbl, feature in zip([df_suvr, df_cds], ['SUVr GLM', 'DI'], ['mean', 'DI']):
            for data, lbl, feature in zip([df_suvr, df_cds], ['SUVr_best', 'DI_x'], ['mean', 'DI']):
                res_df = pd.DataFrame()
                data['Subject'] = data.index
                for voi in data['VOI'].unique():
                #for voi in df_cds['VOI'].unique():  # print only those for which we have DI
                    fig, ax = plt.subplots()
                    data_voi = data[data['VOI'] == voi]

                    ttest_res = data_voi.pairwise_tests(dv=feature,
                                                    between='Cohort',
                                                    subject='Subject',
                                                    #within='VOI',
                                                    effsize='cohen',
                                                    #interaction=False,
                                                    )
                    ttest_res['VOI'] = [voi] * len(ttest_res)
                    res_df = pd.concat([res_df, ttest_res], ignore_index=True)
                    # data_voi.head()
                    # print(ttest_res.loc[ttest_res['Contrast'] == 'Cohort'])
                    # sns.boxplot(data=data_voi, x='Cohort', y=feature, ax=ax, showfliers=False)
                    # sns.stripplot(data=data_voi, x='Cohort', y=feature, ax=ax, color='k', alpha=0.2)
                    # p = ttest_res.loc[ttest_res['Contrast'] == 'Cohort', 'p-unc'].values[0]
                    # effsize = ttest_res.loc[ttest_res['Contrast'] == 'Cohort', 'cohen'].values[0]
                    # ax.set_title(f"{lbl}, {voi},\n"
                    #              f"p = {p:.3f}, cohen's d = {effsize:.2f}")
                    # fig.savefig(f'{out_dir}/{cohort.name}_vs_{ref_cohort.name}_{voi}_{lbl}.png')
                res_df.sort_values(by=['cohen'], inplace=True, ascending=True)
                if lbl == 'SUVr GLM':
                    best_ref_vois[cohort_name] = res_df.iloc[0]['VOI']
                    best_ref_vois[ref_cohort_name] = res_df.iloc[0]['VOI']
                    best_ref_vois_cohen[cohort_name] = res_df.iloc[0]['cohen']
                    best_ref_vois_cohen[ref_cohort_name] = res_df.iloc[0]['cohen']
                res_df.to_excel(f'{out_dir}/{cohort.name}_vs_{ref_cohort.name}_{lbl}.xlsx', index=False)



    #
    #
    #
    #
    #
    # scaling_voi_dict = dict(Onset = 'BS_PVC',
    #                         Moderate = 'R_SSE',
    #                         WT_PLX = 'CB_PVC',
    #                         TREM2 = 'L_THA',
    #                         APPPS1_TREM2 = 'R_AUD',
    #                         deltaE9_PLX = 'BFS',
    #                         PS2APP_PLX = 'BFS',
    #                         )
    #
    #
    #
    # for study_name, ref_cohort_name in zip(study_dfs.keys(), ['WT_02M', 'WT_06M', 'Placebo', 'WT_12M', 'APPPS1_tr_TREM2_wt', 'FU1_TG_Plac', 'FU_TG_Plac']):
    # #for study_name, ref_cohort_name in zip(['Onset'], ['WT_02M']):
    # #for study_name, ref_cohort_name in zip(['Moderate'], ['WT_06M']):
    #     for cohort_name in study_dfs[study_name].columns:
    #     #for cohort_name in ['APPki_02_5M']:
    #     #for cohort_name in ['APPki_05M']:
    #         if cohort_name == ref_cohort_name:
    #             continue
    #         cohort_pair = []
    #         for c_name in [cohort_name, ref_cohort_name]:
    #             cohort_nim_stems = study_dfs[study_name][c_name].to_list()
    #             #cohort_nim_paths = [nim_path for nim_path in filenames if cohort_name in nim_path]
    #             cohort_nim_paths = [f'{data_dir}/{cohort_nim_stem}.nii' for cohort_nim_stem in cohort_nim_stems]
    #             cohort = Cohort(cohort_nim_paths, cohort_name=c_name)
    #             cohort.load_image_arrays()
    #             cohort.scale(atlas=atlas, label=scaling_voi_dict[study_name], feature='mean')
    #             cohort.extract_features(atlas=atlas, feature='mean',
    #                                     save_path=f'{out_dir}/means_mirrione_cohort_{cohort_name}_{scaling_voi_dict[study_name]}_scaling.xlsx')
    #             cohort_pair.append(cohort)
    #
    #         cohort = cohort_pair[0]
    #         ref_cohort = cohort_pair[1]
    #
    #         for coh in [cohort, ref_cohort]:
    #             df_tmp = coh.combined_features_long
    #             df_tmp['Cohort'] = [coh.name] * len(df_tmp)
    #
    #
    #         df_suvr = pd.concat([ref_cohort.combined_features_long, cohort.combined_features_long], ignore_index=False)
    #         #for data, lbl, feature in zip([df_suvr, df_cds], ['SUVr GLM', 'DI'], ['mean', 'DI']):
    #         for data, lbl, feature in zip([df_suvr], ['SUVr_best'], ['mean']):
    #             res_df = pd.DataFrame()
    #             data['Subject'] = data.index
    #             for voi in data['VOI'].unique():
    #             #for voi in df_cds['VOI'].unique():  # print only those for which we have DI
    #                 fig, ax = plt.subplots()
    #                 data_voi = data[data['VOI'] == voi]
    #
    #                 ttest_res = data_voi.pairwise_tests(dv=feature,
    #                                                 between='Cohort',
    #                                                 subject='Subject',
    #                                                 #within='VOI',
    #                                                 effsize='cohen',
    #                                                 #interaction=False,
    #                                                 )
    #                 ttest_res['VOI'] = [voi] * len(ttest_res)
    #                 res_df = pd.concat([res_df, ttest_res], ignore_index=True)
    #                 # data_voi.head()
    #                 # print(ttest_res.loc[ttest_res['Contrast'] == 'Cohort'])
    #                 # sns.boxplot(data=data_voi, x='Cohort', y=feature, ax=ax, showfliers=False)
    #                 # sns.stripplot(data=data_voi, x='Cohort', y=feature, ax=ax, color='k', alpha=0.2)
    #                 # p = ttest_res.loc[ttest_res['Contrast'] == 'Cohort', 'p-unc'].values[0]
    #                 # effsize = ttest_res.loc[ttest_res['Contrast'] == 'Cohort', 'cohen'].values[0]
    #                 # ax.set_title(f"{lbl}, {voi},\n"
    #                 #              f"p = {p:.3f}, cohen's d = {effsize:.2f}")
    #                 # fig.savefig(f'{out_dir}/{cohort.name}_vs_{ref_cohort.name}_{voi}_{lbl}.png')
    #             res_df.sort_values(by=['cohen'], inplace=True, ascending=True)
    #
    #             res_df.to_excel(f'{out_dir}/{cohort.name}_vs_{ref_cohort.name}_{lbl}.xlsx', index=False)