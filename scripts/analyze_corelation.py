#%%
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
from pathlib import Path
from src import plotting

#%%
def fit_model(data,region):
    # fit linear regression model with slope and intercept
    model = sm.OLS(data[region], sm.add_constant(data["chron_age"]))
    results = model.fit()

    # get slope and intercept
    intercept = results.params[0]
    slope = results.params[1]


    return intercept, slope

# %% set data path
RESULTDIR = Path(__file__).parent.parent / "results"
WORKLIST_ALT = RESULTDIR / f"prepared_data_no_tracer_alt.xlsx"
WORKLIST_C36 = RESULTDIR / f"prepared_data_no_tracer_c36.xlsx"

# %% load data
data_alt = pd.read_excel(WORKLIST_ALT, index_col=0)
data_c36 = pd.read_excel(WORKLIST_C36, index_col=0)

#%% plot correlation for chron_age - binding for binding = "frontal", "hippocampus", "caudate"
cm = 1/2.54
plotting.set_r_params(small=8, medium=10, big=12)

fig,axs = plotting.get_figures(1,2, figsize=(14*cm,7*cm))

# plot data
for i, binding in enumerate(["temporal", "hippocampus"]):
    # fit slope
    intercept, slope = fit_model(data_alt, binding)

    # plot scatter
    sns.scatterplot(
        x=data_alt["chron_age"],
        y=data_alt[binding],
        ax=axs[i]
    )

    # plot regression line
    sns.lineplot(x=data_alt["chron_age"], y=intercept + slope * data_alt["chron_age"], ax=axs[i], color="C1")
    sns.despine(ax=axs[i])
    axs[i].set_title(binding)
    axs[i].set_xlabel("Chronological Age [years]")
    axs[i].set_ylabel("BPp")
axs[0].set_yticklabels([0,1.0,2.0,3.0,4.0])
fig = plotting.set_style_ax(fig,axs)
plotting.save_figure(fig, RESULTDIR / "figures" / f"correlation_alt_temp_hip.pdf")

# same for c36
fig,axs = plotting.get_figures(1,2, figsize=(14*cm,7*cm))

# plot data
for i, binding in enumerate(["temporal", "hippocampus"]):
    # fit slope
    intercept, slope = fit_model(data_c36, binding)

    # plot scatter
    sns.scatterplot(
        x=data_c36["chron_age"],
        y=data_c36[binding],
        ax=axs[i]
    )

    # plot regression line
    sns.lineplot(x=data_c36["chron_age"], y=intercept + slope * data_c36["chron_age"], ax=axs[i], color="C1")
    sns.despine(ax=axs[i])
    axs[i].set_title(binding)
    axs[i].set_xlabel("Chronological Age [years]")
    axs[i].set_ylabel("BPnd")
    axs[i].set_yticks([0,0.5,1,1.5,2])
axs[1].set_ylim([-0.1,2.1])

fig = plotting.set_style_ax(fig,axs)
plotting.save_figure(fig, RESULTDIR / "figures" / f"correlation_c36_temp_hipp.pdf")
# %%
