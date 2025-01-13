import numpy as np
import pandas as pd
import pingouin as pg
import nibabel as nib
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
from skimage.measure import marching_cubes
from scipy import stats
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorcet as cc
from PIL import Image

from atlas import Atlas
from connectivity import Network, BetweenNetwork
import functions as f


def plot3d_connectivity_on_template(network,
                                    template_nim,
                                    atlas: Atlas,
                                    azim=-60,
                                    elev=30,
                                    template_threshold=0,
                                    template_alpha=0.05,
                                    template_face_color='white',
                                    template_edge_color='grey',
                                    cmap=cc.cm.coolwarm,
                                    line_alpha=0.5,
                                    node_size=6,
                                    node_color=0.2,
                                    node_alpha=0.5,
                                    vmin=None,
                                    vmax=None,
                                    title=None,
                                    save_dir=None,
                                    ):

    # INPUT CHECK
    if not hasattr(atlas, 'voi_centers_df'):
        atlas.find_voi_centers()
    if not np.array_equal(template_nim.affine, atlas.nim.affine):
        template_nim = f.reorient_to_target(input_nifti=template_nim, target_nifti=atlas.nim)
        warnings.warn('Warning: template for 3D plotting and atlas do not have the same affine, '
                      'your template has been reoriented to the atlas. Check the resulting template-atlas matching visually.')

    template_arr = np.squeeze(template_nim.get_fdata())

    # PLOT TEMPLATE
    fig, ax = plot_3d(template_arr,
                      threshold=template_threshold,
                      alpha=template_alpha,
                      face_color=template_face_color,
                      edge_color=template_edge_color,)
    ax.view_init(azim=azim, elev=elev)

    # PLOT CONNECTING LINES
    df = network.df_no_duplicates

    if vmin is None:
        vmin = network.min
    if vmax is None:
        vmax = network.max
    norm = colors.Normalize(vmin, vmax, clip=True)

    for voi_i in network.vois_rows:
        for voi_j in network.vois_cols:
            value = df.loc[voi_i, voi_j]
            if value != 0:
                color = cmap(norm(value))
                ax.plot(atlas.voi_centers_df.loc[[voi_i, voi_j]]['x'],
                        atlas.voi_centers_df.loc[[voi_i, voi_j]]['y'],
                        atlas.voi_centers_df.loc[[voi_i, voi_j]]['z'],
                        color=color,
                        alpha=line_alpha, )

    # PLOT NODES
    for voi, n_connect in network.n_connect_per_voi.items():
        if n_connect != 0:
            voi_center = atlas.voi_centers_df.loc[voi]
            ax.scatter(voi_center['x'], voi_center['y'], voi_center['z'], 'o',
                       s=node_size*n_connect,
                       color=cmap(node_color),
                       edgecolors='black',
                       alpha=node_alpha,
                       )

    # COLORBAR
    axins = inset_axes(ax, width='2%', height='20%', loc='upper right')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, cax=axins, ticks=[vmin, vmax])
    cb.ax.tick_params(color='black', labelcolor='black', direction='out',
                         left=True, right=False, labelleft=True, labelright=False)

    # MISC
    if title is None:
        ax.set_title(f'{network.name}')
    else:
        ax.set_title(title)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fig.savefig(f'{save_dir}/3D_{title}.png', dpi=300, transparent=True)


def plot_3d(arr, threshold=0, alpha=0.05, face_color='white', edge_color='grey'):
    #p = arr.transpose(2, 1, 0)
    p = arr
    verts, faces, normals, values = marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    # face_color = [0.2, 0.2, 0.2]
    # face_color = [0.0, 0.0, 0.0]
    if edge_color:
        mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    ax.grid(b=None)
    ax.axis('off')
    axisEqual3D(ax=ax)
    del mesh
    del p
    return fig, ax

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def plot_distributions(list_of_data,
                       x='network_name', y='abs_value', hue='value',
                       simplified=True,
                       vmin=None, vmax=None,
                       cmap=cc.cm.coolwarm, hue_vmin=None, hue_vmax=None,
                       xlabels=None, ylabel=None, cbar_label='',
                       save_path=None, save_stats=True, kwargs_stats={}):

    # GET COMBINED DATAFRAME FOR PLOTTING
    df = None
    for i, data in enumerate(list_of_data):
        if isinstance(data, Network) or isinstance(data, BetweenNetwork):
            df_temp = data.df_valid_long
            if xlabels is not None:
                df_temp[x] = [xlabels[i]] * len(df_temp)
            elif list_of_data[i].name != '':
                df_temp[x] = [list_of_data[i].name] * len(df_temp)
            df = pd.concat([df, df_temp], ignore_index=True)
        else:
            raise TypeError('Expected type: Network or BetweenNetwork object')
        if df[x].unique().size == 1 and xlabels is None:
            print('WARNING in plot_distributions: all xlabels have the same name. If you are plotting distribution for a single network, ignore this warning. Otherwise, set distinct xlabels.')

    fig, ax = plt.subplots()

    # BOXPLOT
    sns.boxplot(data=df, x=x, y=y, ax=ax,
                showfliers=False,
                width=0.25,
                boxprops=dict(facecolor=(0, 0, 0, 0),
                              linewidth=3, zorder=3),
                whiskerprops=dict(linewidth=3),
                capprops=dict(linewidth=3),
                medianprops=dict(linewidth=3))

    # VIOLINPLOT
    sns.violinplot(data=df, x=x, y=y, ax=ax,
                   color='lightgray',
                   cut=0, inner=None)
    # ax.tick_params(axis='x', labelrotation=45)
    for item in ax.collections:
        x0, y0, width, height = item.get_paths()[0].get_extents().bounds
        item.set_clip_path(plt.Rectangle((x0, y0), width / 2, height,
                                         transform=ax.transData))

    # COLORCODED STRIPPLOT
    if simplified:
        sns.stripplot(data=df, x=x, y=y, color='k', alpha=0.05, size=1, ax=ax)
    else:
        colorcoded_stripplot(df=df, x=x, y=y, hue=hue, fig=fig, ax=ax,
                             cmap=cmap, vmin=hue_vmin, vmax=hue_vmax, cbar_label=cbar_label)

    # ADDITIONAL SETTINGS
    ax.set_ylim([vmin, vmax])
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    #ax.tick_params(axis='x', labelrotation=45)
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')

    # CALCULATE STATISTICS
    if save_stats:
        if not kwargs_stats:
            stats = df.pairwise_tests(dv='abs_value', within='network_name', subject='VOI_pair', parametric=False)
        else:
            stats = df.pairwise_tests(**kwargs_stats)
        stats.to_excel(f'{os.path.splitext(save_path)[0]}.xlsx', index=False)
    # plt.close()


def colorcoded_stripplot(df, x, y, hue, fig, ax, cmap=cc.cm.coolwarm, vmin=None, vmax=None, cbar_label=''):

    if vmin is None:
        vmin = df[hue].min()
    if vmax is None:
        vmax = df[hue].max()

    from matplotlib import colors
    norm = colors.Normalize(vmin, vmax, clip=True)

    # create a color dictionary (value in c : color from colormap)
    colors = {}
    for cval in df[hue]:
        colors.update({cval: cmap(norm(cval))})

    np.random.seed(123)
    sns.stripplot(x=x, y=y, hue=hue, data=df, palette=colors, ax=ax,
                      linewidth=0.5, jitter=0.05)
    for item in ax.collections:
        item.set_offsets(item.get_offsets() + np.array([0.07, 0]))
    # remove the legend, because we want to set a colorbar instead
    plt.gca().legend_.remove()

    # colorbar
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size='5%', pad=0.05)
    fig.add_axes(ax_cb)
    cb1 = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label(cbar_label)


def plot_matrix(data, data2=None, save_path=None, data_label='', data2_label='', ticks=True, cmap=cc.cm.coolwarm, **kwargs):
    if data2 is not None:
        if not data.index.equals(data2.index):
            raise Exception('Input dataframes do not have the same index')
        matrix1 = data.to_numpy()
        matrix2 = data2.to_numpy()
        matrix = np.tril(matrix1) + np.triu(matrix2)
        data = pd.DataFrame(matrix, index=data.index, columns=data.columns)

    fig, ax = plt.subplots()
    sns.heatmap(data,
                cmap=cmap,
                #linewidths=.5,
                #cbar_kws={'shrink': .5},
                **kwargs)
    if ticks:
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.tick_params(axis='x', labelrotation=90, labelsize=5)
        ax.tick_params(axis='y', labelrotation=0, labelsize=5)
    else:
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False,)
    ax.set_aspect('equal', 'box')
    #ax.set(xlabel=data2_label, ylabel=data_label)
    ax.set_xlabel(data2_label, fontsize=15)
    ax.set_ylabel(data_label, fontsize=15)
    ax.xaxis.set_label_position('top')
    ax.plot([0, 1], [1, 0], 'k-', linewidth=0.5, transform=ax.transAxes)
    plt.tight_layout()

    # rotate the plot 45° if two matrices are given
    if data2 is not None:
        plt.gcf().axes[1].tick_params(rotation=45)  # gcf = "get current figure", axes[1] is the colorbar Axes
        temp_path = './temp.png'
        fig.savefig(temp_path, dpi=400, transparent=True)
        img = Image.open(temp_path)
        img2 = img.rotate(-45, expand=1)
        img2.show()
        os.remove(temp_path)
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))
            img2.save(save_path)
    else:
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=400, transparent=True)


def voi_boxplots(cohorts, attr='extracted_features',
                 subnetwork_name='', p_thr=0.05, p_given_dict=None,
                 palette=None, color='w',
                 save_folder=None, prefix='', return_stats=False,
                 figsize=None,
                 ):

    if palette is not None:
        color = None

    df = pd.DataFrame()
    vois = None
    for cohort in cohorts:
        df_temp = getattr(cohort, attr).copy()
        if subnetwork_name:
            vois = cohort.connectivity.subnetworks[subnetwork_name].vois
            df_temp = df_temp[vois]
        vois = df_temp.columns
        #df_temp = cohort.cds.copy()
        df_temp['Cohort'] = [cohort.name] * len(df_temp)
        df_temp['Subject'] = df_temp.index
        df = pd.concat([df, df_temp], ignore_index=False)

    df_anova = pd.DataFrame()
    df_ttest = pd.DataFrame()
    if prefix: prefix = f'_{prefix}'
    for voi in vois:
        df_anova_temp = df.anova(dv=voi, between='Cohort')
        df_anova_temp['VOI'] = [voi] * len(df_anova_temp)
        df_anova = pd.concat([df_anova, df_anova_temp], ignore_index=True)
        p = df_anova_temp['p-unc'].to_numpy()[0]
        if p_given_dict is not None:
            p_given = p_given_dict[voi]
        else:
            p_given = None

        df_ttest_temp = df.pairwise_tests(dv=voi, between='Cohort', effsize='cohen', padjust='fdr_bh')
        df_ttest_temp['VOI'] = [voi] * len(df_ttest_temp)
        df_ttest = pd.concat([df_ttest, df_ttest_temp], ignore_index=True)

        if p < p_thr and save_folder is not None:
            p_given_suffix = ''
            if p_given is not None:
                if p_given >= p_thr:
                    continue
                else:
                    p_given_suffix = f'_pFDR_{p_given:.4f}_'

            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(x='Cohort', y=voi, data=df, color=color, palette=palette, showfliers=False, ax=ax)
            sns.stripplot(x='Cohort', y=voi, data=df, color='k', alpha=0.5, s=7, ax=ax)
            ax.tick_params(axis='both', labelsize=15)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
            fig.tight_layout()
            fig.savefig(f'{save_folder}/{attr}{prefix}_anova_p_{p:.4f}{p_given_suffix}_voi_{voi}.png', bbox_inches='tight', transparent=True, dpi=200)

    df_anova.set_index('VOI', inplace=True)
    df_anova.sort_values(by='p-unc', ascending=True, inplace=True)
    if p_given_dict is not None:
        df_anova['pFDR'] = df_anova.index.map(p_given_dict)

    df_ttest.set_index('VOI', inplace=True)
    df_ttest.sort_values(by='p-unc', ascending=True, inplace=True)

    if save_folder is not None:
        df_anova.to_excel(f'{save_folder}/{attr}{prefix}_anova.xlsx')
        df_ttest.to_excel(f'{save_folder}/{attr}{prefix}_ttests.xlsx')
    if return_stats:
        return df_anova, df_ttest

def cds_boxplots(cohorts, p_thr=0.05, save_folder=None):

    df = pd.DataFrame()
    vois = cohorts[0].cds.columns

    for cohort in cohorts:
        df_temp = cohort.cds.copy()
        df_temp['Cohort'] = [cohort.name] * len(df_temp)
        df_temp['Subject'] = df_temp.index
        df = pd.concat([df, df_temp], ignore_index=False)

    df_anova = pd.DataFrame()
    for voi in vois:
        df_anova_temp = df.anova(dv=voi, between='Cohort')
        df_anova_temp['VOI'] = [voi] * len(df_anova_temp)
        df_anova = pd.concat([df_anova, df_anova_temp], ignore_index=True)
        p = df_anova_temp['p-unc'].to_numpy()[0]

        if p < p_thr and save_folder is not None:
            fig, ax = plt.subplots()
            sns.boxplot(x='Cohort', y=voi, data=df, showfliers=False, ax=ax)
            sns.stripplot(x='Cohort', y=voi, data=df, color='k', alpha=0.5, s=7, ax=ax)
            ax.tick_params(axis='both', labelsize=15)
            fig.savefig(f'{save_folder}/cds_anova_p_{p:.4f}_voi_{voi}.png', bbox_inches='tight', transparent=True, dpi=200)

    df_anova.set_index('VOI', inplace=True)
    df_anova.sort_values(by='p-unc', ascending=True, inplace=True)
    df_anova.to_excel(f'{save_folder}/cds_anova.xlsx')


def feature_pair_correlation(cohorts, feature_x, feature_y, plot=True,
                             annotate_subjects=False,
                             color='blue', scatter_size=150, scatter_alpha=0.5,
                             display_fit_results=True,
                             loc_fit_results=(0.95, 0.55),
                             align_fit_results='right',
                             xlabel=None, ylabel=None,
                             xlim: tuple | None = None,
                             ylim: tuple | None = None,
                             p_given=None,
                             p_given_suffix='',
                             p_plotting_threshold=1.0,
                             p_rounding_threshold=0.001,
                             labelsize=20,
                             locator_nbins: int | None = 2,  # 2
                             save_folder=None,
                             figsize: tuple = (5.0, 4.5),
                             dpi=200,
                             ):

    # TODO: warn about missing score for a particular subject
    df = pd.DataFrame(columns=[str(feature_x), str(feature_y), 'Cohort'])
    for cohort in cohorts:
        df_temp = cohort.combined_features[[feature_x, feature_y]].copy()
        df_temp.columns = [str(col) for col in df_temp.columns]
        df_temp['Cohort'] = [cohort.name] * len(df_temp)
        df = pd.concat([df, df_temp])
    if df.isnull().values.any():
        warnings.warn('One or more subjects have missing values')
        df = df.dropna()
    feature_x = str(feature_x)
    feature_y = str(feature_y)

    m, b, r, p, stderr = stats.linregress(df[feature_x], df[feature_y])

    if not plot:
        return {'feature_x': feature_x, 'feature_y': feature_y, 'r': r, 'p': p, 'slope': m, 'intercept': b}

    if p_given is not None:
        p_thr = p_given
        p_given_str = f'p{p_given_suffix} = {p_given:.3f}\n'
        p_given_suffix = f'_p{p_given_suffix}{p_given:.3f}'
    else:
        p_thr = p
        p_given_str = ''

    if p_thr < p_plotting_threshold:
        fig, ax = plt.subplots(figsize=figsize)
        sns.regplot(data=df,
                    x=feature_x,
                    y=feature_y,
                    ax=ax,
                    color=color,
                    scatter=False,
                    )
        sns.scatterplot(data=df,
                    x=feature_x,
                    y=feature_y,
                    style='Cohort',
                    ax=ax,
                    color=color,
                    s=scatter_size,
                    alpha=scatter_alpha,)
        if xlabel is None:
            xlabel = feature_x
        if ylabel is None:
            ylabel = feature_y
        ax.set_xlabel(xlabel, fontsize=labelsize, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=labelsize, fontweight='bold')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(False)
        ax.tick_params(axis='both', labelsize=labelsize)
        if display_fit_results:
            if p < p_rounding_threshold:
                p_str = f'p < {p_rounding_threshold}'
            else:
                p_str = f'p = {p:.3f}'
            infos = f'{p_str}\n' \
                    f'{p_given_str}' \
                    f'r = {r:.3f}\n' \
                    f'm = {m:.2f}\n' \
                    f'b = {b:.2f}\n'
            ax.text(*loc_fit_results, s=infos, color='k', ha=align_fit_results, transform=ax.transAxes)

        if annotate_subjects:
            for x, y, name in zip(df[feature_x], df[feature_y], df.index):
                ax.text(x=x, y=y, s=name, color='k')
        plt.locator_params(axis='both', nbins=locator_nbins)
        fig.tight_layout()

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            out_name = f'{save_folder}/corr_{feature_x}_{feature_y}_p{p:.4f}{p_given_suffix}.png'
            fig.savefig(out_name, transparent=True, bbox_inches='tight', dpi=dpi)

    #plt.close()


def effsize_barplot(data,
                    x, y, hue,
                    ylabel='np2 (effect size)',
                    ylim=None,
                    title='',
                    palette=('#cfcfcf', 'gray', '#a6ba70', '#326c3e'),
                    save_path=None,
                    ):
    fig, ax = plt.subplots()
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, palette=palette)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_combined_bar_strip_spaghetti(
        ax,  # Pass the ax object directly
        data: pd.DataFrame,
        cohort_col: str,
        timepoint_col: str,
        subject_col: str,
        signal_col: str,
        error_metric='sd',  # Error metric for barplot ('sd', 'ci', etc.)
        jitter: float = 0  # Set jitter to 0 to align stripplot and spaghetti plot points
):
    """
    Function to plot a combination of barplot, stripplot, and spaghetti plot on a given ax.

    Parameters:
    - ax: matplotlib.axes.Axes
        The matplotlib Axes object where the plot will be drawn.
    - data: pd.DataFrame
        The input data.
    - cohort_col: str
        The column name representing the cohort.
    - timepoint_col: str
        The column name representing the timepoint.
    - subject_col: str
        The column name representing the subject ID.
    - signal_col: str
        The column name representing the signal level.
    - error_metric: str
        Error metric for the barplot ('sd' for standard deviation, 'ci' for confidence interval, etc.).
    - jitter: float
        The jitter applied to stripplot to avoid overlapping points (set to 0 for no jitter).
    """

    # Barplot for average signal levels (set edgecolor to black, and use one color for all bars)
    # sns.barplot(data=data,
    #             x=cohort_col,
    #             y=signal_col,
    #             hue=timepoint_col,
    #             errorbar=error_metric,
    #             dodge=True,
    #             palette=['lightgray'] * len(data[timepoint_col].unique()),  # Same color for all timepoints
    #             edgecolor='black',  # Black outer contour
    #             ax=ax)
    #

    sns.boxplot(data=data,
                x=cohort_col,
                y=signal_col,
                hue=timepoint_col,
                dodge=True,
                #palette=['lightgray'] * len(data[timepoint_col].unique()),  # Same color for all timepoints
                palette=['white', 'lightgray', 'lightskyblue'],
                #edgecolor='black',  # Black outer contour
                showfliers=False,
                ax=ax)

    # Stripplot to show individual data points (black points with alpha=0.5)
    sns.stripplot(data=data,
                  x=cohort_col,
                  y=signal_col,
                  hue=timepoint_col,
                  dodge=True,
                  jitter=jitter,  # No jitter to align the points for the spaghetti plot
                  color="black",  # All points black
                  alpha=0.5,  # Transparency for the points
                  linewidth=1,
                  edgecolor='black',
                  ax=ax)

    # Remove legend
    ax.legend_.remove()

    # Calculate the unique x positions for each combination of cohort and timepoint manually
    unique_cohorts = data[cohort_col].unique()
    unique_timepoints = data[timepoint_col].unique()
    cohort_positions = {cohort: i for i, cohort in enumerate(unique_cohorts)}

    # Calculate dodge correction based on the number of timepoints
    num_timepoints = len(unique_timepoints)
    dodge_correction = np.linspace(-0.25, 0.25, num_timepoints)  # Slightly wider dodge to ensure connection

    # Map the positions for each combination of subject, cohort, and timepoint
    x_positions_map = {}
    for i, row in data.iterrows():
        cohort = row[cohort_col]
        timepoint = row[timepoint_col]
        cohort_pos = cohort_positions[cohort]  # The baseline cohort position
        timepoint_idx = list(unique_timepoints).index(timepoint)
        # Apply dodge correction to center the timepoints around the cohort position
        x_pos = cohort_pos + dodge_correction[timepoint_idx]
        x_positions_map[(row[subject_col], cohort, timepoint)] = (x_pos, row[signal_col])

    # Now draw the lines (spaghetti plot)
    for subject in data[subject_col].unique():
        subject_data = data[data[subject_col] == subject]

        # Group by cohort to ensure we draw lines only within each cohort
        for cohort in subject_data[cohort_col].unique():
            cohort_data = subject_data[subject_data[cohort_col] == cohort]

            # Sort cohort_data by timepoint for proper connection of lines
            cohort_data = cohort_data.sort_values(by=timepoint_col)

            # Extract the x and y positions for the subject in this cohort
            x_positions = []
            y_positions = []
            for _, row in cohort_data.iterrows():
                x, y = x_positions_map[(row[subject_col], row[cohort_col], row[timepoint_col])]
                x_positions.append(x)
                y_positions.append(y)

            # Draw the spaghetti line connecting the points
            if x_positions and y_positions:
                ax.plot(x_positions, y_positions, color='black', alpha=0.5, linewidth=0.8)

    # Adjusting the position and alignment of timepoint labels
    for i, timepoint in enumerate(unique_timepoints):
        for cohort in unique_cohorts:
            cohort_pos = cohort_positions[cohort]
            timepoint_x_pos = cohort_pos + dodge_correction[i]
            #ax.text(timepoint_x_pos, ax.get_ylim()[0] + 0.05, timepoint, ha='center', va='bottom', rotation=90)

    # Titles and labels
    #ax.set_title("Signal Levels across Cohorts and Timepoints with Subject Connections")
    ax.set_xlabel(None)
    ax.set_ylabel(signal_col)
    ax.tick_params(axis='x', rotation=45)
    #ax.set_ylim(0.5, None)
    ax.figure.tight_layout()






