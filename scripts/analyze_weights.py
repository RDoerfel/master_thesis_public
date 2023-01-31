#%%
from pathlib import Path
import pandas as pd
import seaborn as sns
from src import plotting
import numpy as np
#%%
result_dir = Path(__file__).parent.parent / "results"
pipelines = ["gpr", "svr", "bayesridge", "rvm"]

#%%
def load_weights(result_dir, pipelines):
    """Load weights from pipeline results"""
    weights = {}
    for pipe in pipelines:
        weights[pipe] = pd.read_csv(
            result_dir / f"pred_age_weights_{pipe}.csv")
    return weights

#%% load weights
model = "rvm"
pipelines = [ f"{model}_no_tracer", f"pyment_{model}_no_tracer"]
weights = load_weights(result_dir, pipelines)
titles = ["PET", "PET + MRI"]
#plot weights for rvr and bayesridge
plotting.set_r_params(small=8, medium=10, big=12)
cm = 1/2.54

fig, ax = plotting.get_figures(1, 2, figsize=(15*cm,8*cm))
for i, pipe in enumerate(pipelines):
    pipe_weights = weights[pipe]

    # plot weights
    sns.boxplot(data=pipe_weights, ax=ax[i], orient="h", color="C1", width=0.7, linewidth=0.7, fliersize=2)
    # title
    ax[i].set_title(titles[i])

name_mapping = {
    "Frontal": "Frontal Lobe",
    "Temporal": "Temporal Lobe",
    "Parietal": "Parietal Lobe",
    "Occipital": "Occipital Lobe",
    "Pyment": "pyment"
}
rois = [k.title() for k in pipe_weights.columns]
rois = [name_mapping.get(k, k) for k in rois]
ax[0].set_yticklabels(rois)
fig = plotting.set_style_ax(fig,np.array([ax]))
plotting.save_figure(fig, result_dir / "figures" / f"weights_{model}_no_tracer.pdf")


# %%
