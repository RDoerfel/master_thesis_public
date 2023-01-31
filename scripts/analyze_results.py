# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse
from sklearn.metrics import mean_absolute_error, r2_score

from src import plotting
from src.statistics import CorrelatedTTest


#%% set paths
# path to the results
RESULTDIR = Path(__file__).parent.parent / "results"
DATAFILE = Path(__file__).parent.parent / "data" / "derivatives" / "combined_alt_cimbi_bpnd_fslobes_pyment.xlsx"
PIPELINES = ["bayesridge", "rvm", "gpr", "svr"]

#%% functions
def load_results(result_dir, pipelines):
    """Load results from pipeline results"""
    results = {}
    for pipe in pipelines:
        results[pipe] = pd.read_csv(
            result_dir / f"pred_age_results_{pipe}.csv", index_col='index'
        )
    return results

def load_params(result_dir, pipelines):
    """Load results from pipeline results"""
    results = {}
    for pipe in pipelines:
        data = pd.read_csv(
            result_dir / f"pred_age_best_params_{pipe}.csv"
        )
        params = data['0'].to_list()
        params = [eval(a) for a in params]
        results[pipe] = pd.DataFrame(params)
    return results

def load_mr_brain_age(data_file):
    """Load MR brain age"""
    data = pd.read_excel(data_file, index_col=0)
    mr_brain_age = data["pyment"]
    return mr_brain_age

def load_scores(result_dir, pipelines):
    """Load scores from pipeline results"""
    scores = {}
    for pipe in pipelines:
        scores[pipe] = pd.read_csv(
            result_dir / f"pred_age_scores_{pipe}.csv", index_col=0
        )
    return scores

def load_weights(result_dir, pipelines):
    """Load weights from pipeline results"""
    weights = {}
    for pipe in pipelines:
        weights[pipe] = pd.read_csv(
            result_dir / f"pred_age_weights_{pipe}.csv", index_col=0
        )
    return weights

def append_brain_age(results,mr_brainage):
    """Append brain age to results 
    Args:
        results (dict): dictionary with results for each pipeline
    Returns:
        results (dict): dictionary with pyment appendes as pipeline
    """
    # init dataframe with index of first pipeline
    first_pipe = list(results.keys())[0]
    df = pd.DataFrame(index=results[first_pipe].index)
    df['pred'] = mr_brainage
    df['true'] = results[first_pipe]['true']
    results['pyment'] = df

    return results

#%%
# append pyment_ to pipelines to indicate estimates with pyment as covariate
suffix = "no_tracer"
pipelines = PIPELINES
result_dir = RESULTDIR
data_file = DATAFILE

if suffix:
    suffix = "_" + suffix
    pipelines = [s + suffix for s in pipelines]

# analyze pyment results as well
pipelines = pipelines + ["pyment_" + s for s in pipelines]

# load results
results = load_results(result_dir, pipelines)

# load MR brain age
mr_brain_age = load_mr_brain_age(data_file)

# append brainage to results
results = append_brain_age(results, mr_brain_age)

# add pyment to pipelines
pipelines = pipelines + ["pyment"]

# get list of predictions
predictions = [results[pipe]['pred'] for pipe in pipelines]

# convert to dataframe 
df_pred = pd.concat(predictions, axis=1)
df_pred.columns = pipelines

# append with true age, fold and pyment (same for all pipelines)
df_pred['true'] = results[pipelines[0]]['true']
df_pred['fold'] = results[pipelines[0]]['fold']

# compute pad 
for pipe in pipelines:
    df_pred[f'pad_{pipe}'] = df_pred[pipe] - df_pred.true

# init dataframe for scores per fold
folds = df_pred.fold.unique()
scores = pd.DataFrame(index=folds)

# compute scores for each fold
for fold in df_pred.fold.unique():
    # select data for fold
    df_fold = df_pred[df_pred.fold == fold]
    # compute scores
    for pipe in pipelines:
        # compute MAE
        scores.loc[fold, f'{pipe}_mae'] = mean_absolute_error(df_fold.true, df_fold[pipe])
        # compute R2
        scores.loc[fold, f'{pipe}_r2'] = r2_score(df_fold.true, df_fold[pipe])
        # compute correlation with true
        scores.loc[fold, f'{pipe}_corr'] = df_fold[pipe].corr(df_fold.true)
        # compute correlation with pad and true
        scores.loc[fold, f'{pipe}_corr_pad'] = df_fold[f'pad_{pipe}'].corr(df_fold.true)
    # cound elements
    scores.loc[fold, f'count'] = len(df_fold)

# sort scores into presentable table with pipelines as index and scores as columns

#%%
# convert mae scores into long format for plotting
mae_names = [f'{pipe}_mae' for pipe in pipelines]
scores_mae = scores[mae_names]
scores_mae = scores_mae.reset_index()
scores_mae = scores_mae.melt(id_vars='index', value_vars=mae_names, var_name='pipeline', value_name='mae')

# remove suffix from pipeline name
scores_mae['pipeline'] = scores_mae['pipeline'].str.replace(f'{suffix}', '')

# substract mean of pyment to better see improvement
scores_mae['pipeline'] = scores_mae['pipeline'].str.replace('_mae', '')

# add column with group. PET, PET + pyment and pyment
scores_mae['group'] = "pet"
scores_mae.loc[scores_mae['pipeline'].str.startswith("pyment_"), 'group'] = "pet_mr"
scores_mae.loc[scores_mae['pipeline'] == 'pyment', 'group'] = "mr"
scores_mae.loc[~scores_mae['pipeline'].str.startswith("pyment"), 'group'] = "pet"

mae_pyment = scores_mae.loc[scores_mae['pipeline']=='pyment']['mae'].mean()
#scores_mae['mae'] = scores_mae['mae'] - mae_pyment

# plot stripplot of MAE per fold with 3 subfigures
plotting.set_r_params(small=8,medium=10,big=12)
cm = 1/2.54

fig, ax = plotting.get_figures(1,1, figsize=(15*cm,10*cm),sharex=False, sharey=False) 

# color palette
palette = sns.color_palette('Set2', 3)
colors_b = [palette[0], palette[0], palette[0],palette[0], palette[0], palette[1], palette[1], palette[1], palette[1], palette[1], palette[2]] 
palette = sns.color_palette('Dark2', 3)
colors_s = [palette[0], palette[0], palette[0],palette[0], palette[0], palette[1], palette[1], palette[1], palette[1], palette[1], palette[2]] 

order = ['bayesridge', 'rvm', 'gpr', 'svr', 'space', 'pyment_bayesridge', 'pyment_rvm', 'pyment_gpr', 'pyment_svr', 'space', 'pyment']
sns.stripplot(data=scores_mae, x='mae', y='pipeline', ax=ax, jitter=0.2, size=4, alpha=0.4 ,dodge=True, order=order, palette=colors_s)
sns.boxplot(data=scores_mae, x='mae', y='pipeline', showfliers=False, dodge=True, ax=ax, order=order, width=0.5, color='C7', linewidth=1.5, palette=colors_b)

ax.set_ylabel('')
ax.set_yticklabels(['bayesridge', 'rvm', 'gpr', 'svr', '', 'pyment_bayesridge', 'pyment_rvm', 'pyment_gpr', 'pyment_svr', '', 'pyment'])

ax.set_xlabel('Mean absolute error [years]')

ylim = ax.get_ylim()

ax.vlines(mae_pyment,ymin=ylim[0], ymax=ylim[1], linestyles='dashed', colors='C7')
fig = plotting.set_style_ax(fig,np.array([ax]))
plotting.save_figure(fig, result_dir / "figures" / "mae_comparison.pdf")
# %% plot predicted vs chron age
results_run1 = df_pred[df_pred['fold'].isin([1,2,3,4,5])]

# plot predicted vs chron age

fig, ax = plotting.get_figures(1,1, figsize=(7*cm,7*cm),sharex=False, sharey=False)

sns.scatterplot(ax=ax,data=results_run1,x='true',y='pyment', color='C0', alpha=0.7, s=10, label='pyment')
sns.scatterplot(ax=ax,data=results_run1,x='true',y='pyment_rvm_no_tracer', color='C1', alpha=0.7, s=10, label='pyment_rvm_no_tracer')

ax.set_xlabel('Chronological age [years]')
ax.set_ylabel('Predicted age [years]')
ax.plot([0,100],[0,100], color='C7', linestyle='dashed', linewidth=1)
ax.legend()
ax.set_xlim(-1,101)
ax.set_ylim(-1,101)
ax.set_xticks([0,25,50,75,100])
ax.set_yticks([0,25,50,75,100])

fig = plotting.set_style_ax(fig,np.array([ax]))

