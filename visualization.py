import numpy as np
import pandas as pd
import pingouin as pg
import nibabel as nib
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorcet as cc
from PIL import Image

from atlas import Atlas
from connectivity import Network, BetweenNetwork


def plot3d_connectivity_on_template(network,
                                    template_nim,
                                    atlas: Atlas,
                                    azim=-70,
                                    elev=0,
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
    assert np.array_equal(template_nim.affine, atlas.nim.affine), 'template and atlas have different spacings'
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
    colorcoded_stripplot(df=df, x=x, y=y, hue=hue, fig=fig, ax=ax,
                         cmap=cmap, vmin=hue_vmin, vmax=hue_vmax, cbar_label=cbar_label)

    # ADDITIONAL SETTINGS
    ax.set_ylim([vmin, vmax])
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=45)
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=300, transparent=True)

    # CALCULATE STATISTICS
    if save_stats:
        if not kwargs_stats:
            stats = df.pairwise_tests(dv='abs_value', within='network_name', subject='VOI_pair', parametric=False)
        else:
            stats = df.pairwise_tests(**kwargs_stats)
        stats.to_excel(f'{os.path.splitext(save_path)[0]}.xlsx', index=False)
    plt.close()


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







