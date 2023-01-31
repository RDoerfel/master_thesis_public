#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
from scipy import stats
from pathlib import Path

from src import fs_roi_lut
from src import plotting
#%% plotting
plotting.set_r_params(small=8,medium=10,big=12)
cm = 1/2.54

#%% Load data
result_dir = Path(__file__).parent.parent / "results" 
figure_dir = result_dir / "figures"
data_file = result_dir / "prepared_data_no_tracer.xlsx"
data = pd.read_excel(data_file, index_col=0)

#%% read rois
roi_file = Path(__file__).parent.parent / "data" / "rois_fs.json"
rois = fs_roi_lut.read_roi_names(roi_file)
rois = [item for item in rois if not "cerebellum" in item]
rois = [item for item in rois if not "brain-stem" in item]

name_mapping = {
    "frontal": "Frontal Lobe",
    "temporal": "Temporal Lobe",
    "parietal": "Parietal Lobe",
    "occipital": "Occipital Lobe"
}
# %% correlate rois with age using scipy.stats.pearsonr
corrs_c = []
corrs_c_ci = []
corrs_a = []
corrs_a_ci = []
for roi in rois:
    pears_c = stats.pearsonr(data.loc[data['tracer_C']==1][roi], data["chron_age"].loc[data['tracer_C']==1])
    corrs_c.append(pears_c[0])
    ci_c = pears_c.confidence_interval() 
    corrs_c_ci.append([ci_c[0], ci_c[1]])
    pears_a = stats.pearsonr(data.loc[data['tracer_a']==1][roi], data["chron_age"].loc[data['tracer_a']==1])
    corrs_a.append(pears_a[0])
    ci_a = pears_a.confidence_interval()
    corrs_a_ci.append([ci_a[0], ci_a[1]])

# concatenate all into one dataframe with corr and corr_ci
corrs_c = pd.DataFrame(corrs_c, columns=['correlation'])
corrs_c_ci = pd.DataFrame(corrs_c_ci, columns=['lower', 'upper'])

corrs_a = pd.DataFrame(corrs_a, columns=['correlation'])
corrs_a_ci = pd.DataFrame(corrs_a_ci, columns=['lower', 'upper'])

corrs_c = pd.concat([corrs_c, corrs_c_ci], axis=1)
corrs_c['roi'] = rois
corrs_c['tracer'] = 'C36'
corrs_a = pd.concat([corrs_a, corrs_a_ci], axis=1)
corrs_a['tracer'] = 'Alt'
corrs_a['roi'] = rois

# combine correlation results in long table but keep correlation, lower and upper´
corrs = pd.concat([corrs_c, corrs_a], axis=0)
corrs.reset_index(inplace=True, drop=True)
#%%

# combine correlation results in long table but keeping index
#corrs = pd.concat([corrs_c, corrs_a], axis=1, keys=['C36', 'Alt'])
#corrs = corrs.stack().reset_index()
#corrs.columns = ['roi', 'tracer', 'correlation']
corrs['roi'] = corrs['roi'].replace(name_mapping,regex=True)
corrs['roi'] = [k.title() for k in corrs['roi']]

a = np.ones([2,14])*0.5
# barplot
plotting.set_r_params(small=6, medium=8, big=10)
palette_greys = sns.color_palette("Greys")
greys = [palette_greys[2], palette_greys[5]]
fig, axs = plotting.get_figures(1,1,figsize=(8*cm,7*cm))
yerr=np.row_stack([corrs['lower'], corrs['upper']])

rois_label = corrs['roi'].unique()
# inverse order
rois_label = rois_label[::-1]
for i, roi in enumerate(rois_label):
    # alt data
    alt = corrs.loc[(corrs['roi']==roi) & (corrs['tracer']=='Alt')]
    # c36 data
    c36 = corrs.loc[(corrs['roi']==roi) & (corrs['tracer']=='C36')]
    # get xerr
    xerrc36 = np.row_stack([c36['correlation'] - c36['lower'], c36['upper'] - c36['correlation']])
    xerralt = np.row_stack([alt['correlation'] - alt['lower'], alt['upper'] - alt['correlation']])

    # add point for each at i +- 0.2
    axs.errorbar(y=i+0.2, x=c36['correlation'], xerr=xerrc36, fmt='o', color=greys[1], label='C36', linewidth=1, markersize=3)
    axs.errorbar(y=i-0.2, x=alt['correlation'], xerr=xerralt, fmt='o', color=greys[0], label='Alt', linewidth=1, markersize=3)

axs.vlines(0, -0.4, 13.4, linestyles='dashed', linewidth=0.5, color='C3')
axs.set_yticks(np.arange(len(rois_label)))
axs.set_yticklabels(rois_label)
axs.set_ylim([-0.4, 13.4])

axs.set_xlabel('Correlation with age')
axs.set_ylabel('Region of Interest')
fig = plotting.set_style_ax(fig,np.array([axs]))
fig.savefig(figure_dir / 'data_roi_correlation_age.pdf', dpi=300)

#%% barplots mean/sd for rois
data_long = data.melt(id_vars=['tracer_a'], value_vars=rois, var_name='roi', value_name='value')

# compute mean and sd for each roi
data_long = data_long.groupby(['roi', 'tracer_a']).agg({'value': ['mean', 'std']})
data_long = data_long.reset_index()
data_long.columns = ['roi', 'tracer_a', 'mean', 'sd']

name_mapping = {
    "Frontal": "Frontal Lobe",
    "Temporal": "Temporal Lobe",
    "Parietal": "Parietal Lobe",
    "Occipital": "Occipital Lobe"
}
data_long['roi'] = [k.title() for k in data_long['roi']]
data_long['roi'] = data_long['roi'].replace(name_mapping,regex=True)

# plot bars using matplotlib
y_pos = []
height = []
xerr = []
for i, roi in enumerate(rois_label):
    # alt data
    alt = data_long.loc[(data_long['roi']==roi) & (data_long['tracer_a']==0)]

    height.append(alt['mean'].to_numpy()[0])
    # c36 data
    c36 = data_long.loc[(data_long['roi']==roi) & (data_long['tracer_a']==1)]
    height.append(c36['mean'].to_numpy()[0])
    # get xerr
    xerr.append(c36['sd'].to_numpy()[0])
    xerr.append(alt['sd'].to_numpy()[0])
    # add point for each at i +- 0.2
    y_pos.append(i-0.2)
    y_pos.append(i+0.2) 

fig, axs = plotting.get_figures(1,1,figsize=(8*cm,7*cm))

axs.barh(y=y_pos, align="center", width=height, color=greys, height=0.4, xerr=xerr, linewidth=0.2)

axs.set_xlabel('ROI mean')
axs.set_ylabel('Region of Interest')
axs.set_yticks(np.arange(len(rois_label)))
axs.set_yticklabels(rois_label)
axs.set_ylim([-0.4, 13.4])
fig = plotting.set_style_ax(fig,np.array([axs]))
fig.savefig(figure_dir / 'data_roi_mean.pdf', dpi=300)


# %%

def plot_heatmap(data):
    connectivity = data.corr()
    mask = np.tril(connectivity)
    np.fill_diagonal(mask, False)
    cm = 1/2.54
    plotting.set_r_params(small=6, medium=8, big=10)
    fig, axs = plotting.get_figures(1,1,figsize=(7*cm,7*cm))
    cbar_kws = {'label': "Pearson's r", "location": "bottom", "use_gridspec": False}
    sns.heatmap(connectivity, cmap='coolwarm', ax=axs,mask=mask,vmin=0, vmax=1, linewidths=0.5, square=True,cbar_kws=cbar_kws)
    axs.set_xticklabels(data.columns, rotation=90)
    axs.set_yticklabels(data.columns, rotation=0)
    axs.xaxis.tick_top()

    return fig, axs

# %% plot connectivity for each tracer
data_a = data.loc[data['tracer_a']==1][rois]
data_c = data.loc[data['tracer_C']==1][rois]

data_a.rename(columns=name_mapping, inplace=True)
data_c.rename(columns=name_mapping, inplace=True)

# make first letter uppercase in columns
name_mapping = {k: k.title() for k in data_a.columns}
data_a.rename(columns=name_mapping, inplace=True)
data_c.rename(columns=name_mapping, inplace=True)

# rename columns for plotting
data_a.rename(columns=name_mapping, inplace=True)
data_c.rename(columns=name_mapping, inplace=True)
fig, axs = plot_heatmap(data_c)
fig.savefig(figure_dir / 'data_roi_connectivity_C.pdf', dpi=300)
fig, axs = plot_heatmap(data_a)
fig.savefig(figure_dir / 'data_roi_connectivity_a.pdf', dpi=300)

# %%
