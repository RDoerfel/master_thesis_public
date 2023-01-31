#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.model_selection import train_test_split
from statsmodels.distributions.empirical_distribution import ECDF

from src import nodim
from src import prep
from src import plotting
from src.fs_roi_lut import read_roi_names
#%% first simulate dat
CM = 1/2.54

# simulate data

def run_nodim_simulated(data1, data2, name):
    params1 = nodim.estimate_params(data1)
    params2 = nodim.estimate_params(data2)

    # compute pdfs
    x = np.linspace(-7.5,7.5, 1000)
    pdf1 = stats.norm.pdf(x, loc = params1['mu'], scale = params1['std'])
    pdf2 = stats.norm.pdf(x, loc = params2['mu'], scale = params2['std'])

    # convert data1 to data2
    data1_transformed = nodim.transform_nodim(params1, params2, data1)
    params_transformed = nodim.estimate_params(data1_transformed)

    # run ks test
    ks = stats.ks_2samp(data1_transformed, data2)
    print(ks)

    # compute pdfs
    pdf1_transformed = stats.norm.pdf(x, loc = params_transformed['mu'], scale = params_transformed['std'])

    # plot
    plotting.set_r_params(small=8,medium=10,big=12)
    fig, axs = plotting.get_figures(1,2,figsize=(15*CM,5*CM),sharey=True,sharex=True)

    sns.histplot(data1, ax=axs[0], label='data1',stat='density',color='C0')
    sns.histplot(data2, ax=axs[0], label='data2',stat='density',color='C1')
    axs[0].plot(x, pdf1, label='pdf1',color='C0')
    axs[0].plot(x, pdf2, label='pdf2',color='C1')
    axs[0].set_title('original data')

    sns.histplot(data1_transformed, ax=axs[1], label='data1 transformed',stat='density',color='C0')
    sns.histplot(data2, ax=axs[1], label='data2',stat='density',color='C1')
    axs[1].plot(x, pdf1_transformed, label='pdf1 transformed',color='C0')
    axs[1].plot(x, pdf2, label='pdf2',color='C1')
    axs[1].set_title('transformed data')


    for ax in axs:
        ax.set_xlabel('x')
        ax.set_ylabel('pdf')
        ax.set_yticks([0,0.2,0.4])

    fig = plotting.set_style_ax(fig,axs)
    #fig.savefig(Path(__file__).parent.parent / "results" / "figures" / f'nodim_pdf_cdf_simulated_{name}.png', dpi=300)

mu1 = 0
std1 = 1.5
data1 = np.random.normal(loc = mu1, scale = std1, size = 1000)
data2 = data1 * 1.5 + 2

run_nodim_simulated(data1, data2, 'samepop')
mu2 = 2
std2 = 3
data2 = np.random.normal(loc = mu2, scale = std2, size = 1000)
run_nodim_simulated(data1, data2, 'diffpop')


#%% load data
data_file = Path(__file__).parent.parent / "data" / "derivatives" / "combined_alt_cimbi_bpnd_fslobes_pyment.xlsx"
data = pd.read_excel(data_file)

roi_file = Path(__file__).parent.parent / "data" / "rois_fs.json"
rois = read_roi_names(roi_file)
rois = [item for item in rois if not "cerebellum" in item]

# drop HRRT
data = data[
    ((data["tracer"] == "a") & (data["camera"] == "GE")) | (data["tracer"] == "C")
]

data = prep.bin_data(data, "chron_age", min_val=0.0, max_val=100.0, step=10)

# split data 50/50 stratified by chron_age_group
data_train, data_test = train_test_split(data, test_size=0.5, stratify=data['tracer'])
print("Tracer in train data:\n",data_train['tracer'].value_counts())
print("Tracer in test data:\n", data_test['tracer'].value_counts())


# %% plot distributions of all rois for c36 and alt 
rois_to_use = ['temporal', 'hippocampus', 'caudate']
for roi in rois_to_use:
    roi_c36 = data[data['tracer']=='C'][roi]
    roi_alt = data[data['tracer']=='a'][roi]

    paramc36 = nodim.estimate_params(roi_c36)
    paramalt = nodim.estimate_params(roi_alt)

    # compute pdfs
    x = np.linspace(data[roi].min() -0.2, data[roi].max()+0.2, 1000)
    pdfc36 = stats.norm.pdf(x, loc = paramc36['mu'], scale = paramc36['std'])
    pdfalt = stats.norm.pdf(x, loc = paramalt['mu'], scale = paramalt['std'])

    # compute cdfs
    cdfc36 = stats.norm.cdf(x, loc = paramc36['mu'], scale = paramc36['std'])
    cdfalt = stats.norm.cdf(x, loc = paramalt['mu'], scale = paramalt['std'])

    # estimate empirical cdf
    ecdf = ECDF(roi_c36)
    ecdfalt = ECDF(roi_alt)

    # transform data
    c36_transformed = nodim.transform_nodim(paramc36, paramalt, roi_c36)
    paramc36_transformed = nodim.estimate_params(c36_transformed)
    pdfc35_transformed = stats.norm.pdf(x, loc = paramc36_transformed['mu'], scale = paramc36_transformed['std'])
    cdfc36_transformed = stats.norm.cdf(x, loc = paramc36_transformed['mu'], scale = paramc36_transformed['std'])
    ecdf_transformed = ECDF(c36_transformed)

    # kstest
    kstest_c36 = stats.ks_2samp(c36_transformed, roi_alt)

    # plot
    plotting.set_r_params(small=8,medium=10,big=12)
    fig, axs = plotting.get_figures(3,1,figsize=(5*CM,15*CM), sharex=True, sharey=False)

    # plot data
    sns.histplot(roi_c36, ax=axs[0], label='c36',stat='density',color='C0')
    sns.histplot(roi_alt, ax=axs[0], label='alt',stat='density',color='C1')
    sns.histplot(c36_transformed, ax=axs[0], label='c36 trans',stat='density',color='C2',linestyle='--')
    axs[0].set_title(f'{roi} data')
    axs[0].set_ylabel('density')
    axs[0].legend()

    # plot pdf
    axs[1].plot(x, pdfc36, label='pdfc36',color='C0')
    axs[1].plot(x, pdfalt, label='pdfalt',color='C1')
    axs[1].plot(x, pdfc35_transformed, label='pdfc36 trans',color='C2',linestyle='--')
    axs[1].set_title(f'{roi} pdf')
    axs[1].set_ylabel('pdf')
    axs[1].legend()

    # plot cdf
    axs[2].plot(x, cdfc36, label='cdfc36',color='C0')
    axs[2].plot(x, cdfalt, label='cdfalt',color='C1')
    axs[2].plot(x, ecdf(x), label='ecdfc36',color='C0',linestyle='--')
    axs[2].plot(x, ecdfalt(x), label='ecdfalt',color='C1',linestyle='--')
    axs[2].plot(x, cdfc36_transformed, label='cdfc36 trans',color='C2')
    axs[2].plot(x, ecdf_transformed(x), label='ecdfc36 trans',color='C2',linestyle='-.')
    axs[2].set_title(f'{roi} cdf')
    axs[2].set_ylabel('cdf')
    axs[2].set_xlabel('binding potential')
    axs[2].legend()

    fig = plotting.set_style_ax(fig,axs)
    #fig.savefig(Path(__file__).parent.parent / "results" / "figures" / f'nodim_pdf_cdf_{roi}.png', dpi=300)

# %%
def plot_dimap_for_roi(data1, data2, roi, ylim, yticks):
    paramc36 = nodim.estimate_params(data1)
    paramalt = nodim.estimate_params(data2)

    # compute pdfs
    min_x = min(data1.min(), data2.min())
    max_x = max(data1.max(), data2.max())
    x = np.linspace(min_x -0.2, max_x+0.2, 1000)
    pdfc36 = stats.norm.pdf(x, loc = paramc36['mu'], scale = paramc36['std'])
    pdfalt = stats.norm.pdf(x, loc = paramalt['mu'], scale = paramalt['std'])

    # compute cdfs
    cdfc36 = stats.norm.cdf(x, loc = paramc36['mu'], scale = paramc36['std'])
    cdfalt = stats.norm.cdf(x, loc = paramalt['mu'], scale = paramalt['std'])

    # estimate empirical cdf
    ecdf = ECDF(data1)
    ecdfalt = ECDF(data2)

    # transform data
    c36_transformed = nodim.transform_nodim(paramc36, paramalt, data1)
    paramc36_transformed = nodim.estimate_params(c36_transformed)
    pdfc35_transformed = stats.norm.pdf(x, loc = paramc36_transformed['mu'], scale = paramc36_transformed['std'])
    cdfc36_transformed = stats.norm.cdf(x, loc = paramc36_transformed['mu'], scale = paramc36_transformed['std'])
    ecdf_transformed = ECDF(c36_transformed)

    # ks2 test
    kstest_c36 = stats.ks_2samp(c36_transformed, data2)
    print(f'{roi} ks2 test: {kstest_c36}')

    # plot
    plotting.set_r_params(small=8,medium=10,big=12)
    fig, axs = plotting.get_figures(3,1,figsize=(7.5*CM,15*CM), sharex=True, sharey=False)
    hist_kws=dict(edgecolor="white", linewidth=.5)

    ymin = ylim[0]
    ymax = ylim[1]
    yrange = ymax-ymin
    ymin = ymin - 0.1*yrange
    ymax = ymax + 0.1*yrange

    ylim = [ymin,ymax]

    # plot data
    sns.histplot(data1, ax=axs[0], label='c36',stat='density',color='C0',**hist_kws)
    sns.histplot(data2, ax=axs[0], label='alt',stat='density',color='C1',**hist_kws)
    sns.histplot(c36_transformed, ax=axs[0], label='c36 trans',stat='density',color='C2',**hist_kws)
    axs[0].set_ylabel('density')
    axs[0].set_yticks(yticks)
    axs[0].set_ylim(ylim)

    # plot pdf
    axs[1].plot(x, pdfc36, label='pdfc36',color='C0')
    axs[1].plot(x, pdfalt, label='pdfalt',color='C1')
    axs[1].plot(x, pdfc35_transformed, label='pdfc36 trans',color='C2',linestyle='--')
    axs[1].set_ylabel('pdf')
    axs[1].set_yticks(yticks)
    axs[1].set_ylim(ylim)

    # plot cdf
    axs[2].plot(x, cdfc36, label='cdfc36',color='C0')
    axs[2].plot(x, cdfalt, label='cdfalt',color='C1')
    axs[2].plot(x, ecdf(x), label='ecdfc36',color='C0',linestyle='--')
    axs[2].plot(x, ecdfalt(x), label='ecdfalt',color='C1',linestyle='--')
    axs[2].plot(x, cdfc36_transformed, label='cdfc36 trans',color='C2')
    axs[2].plot(x, ecdf_transformed(x), label='ecdfc36 trans',color='C2',linestyle='-.')
    axs[2].set_ylabel('cdf')
    axs[2].set_xlabel('binding potential')
    axs[2].set_yticks([0,0.5,1.0])


    fig = plotting.set_style_ax(fig,axs)
    #fig.savefig(Path(__file__).parent.parent / "results" / "figures" / f'nodim_pdf_cdf_{roi}.pdf', dpi=300)

# simulation 1
mu1 = 0
std1 = 1.5
data1 = np.random.normal(loc = mu1, scale = std1, size = 1000)
data2 = data1 * 1.5 + 2

plot_dimap_for_roi(data1, data2, 'Sim1',ylim=[0,0.2],yticks=[0,0.1,0.2])

# simulation 2
mu1 = 0
std1 = 1.5
data1 = np.random.normal(loc = mu1, scale = std1, size = 1000)
mu2 = 2
std2 = 3
data2 = np.random.normal(loc = mu2, scale = std2, size = 1000)

plot_dimap_for_roi(data1, data2, 'Sim2',ylim=[0,0.2],yticks=[0,0.1,0.2])

# 'temporal'
roi = 'temporal'
data_c = data[data['tracer']=='C'][roi]
data_alt = data[data['tracer']=='a'][roi]
plot_dimap_for_roi(data_c, data_alt, 'Temporal',ylim=[0,2],yticks=[0,1,2])

# 'hippocampus'
roi = 'hippocampus'
data_c = data[data['tracer']=='C'][roi]
data_alt = data[data['tracer']=='a'][roi]

plot_dimap_for_roi(data_c, data_alt, 'Hipp.',ylim=[0,4],yticks=[0,2,4])

# Caudate
roi = 'caudate'
data_c = data[data['tracer']=='C'][roi]
data_alt = data[data['tracer']=='a'][roi]

plot_dimap_for_roi(data_c, data_alt, 'Caudate',ylim=[0,5],yticks=[0,2.5,5])
# %%

def run_dimap_ks_test(data1, data2):
    paramc36 = nodim.estimate_params(data1)
    paramalt = nodim.estimate_params(data2)

    # compute pdfs
    min_x = min(data1.min(), data2.min())
    max_x = max(data1.max(), data2.max())
    x = np.linspace(min_x -0.2, max_x+0.2, 1000)
    pdfc36 = stats.norm.pdf(x, loc = paramc36['mu'], scale = paramc36['std'])
    pdfalt = stats.norm.pdf(x, loc = paramalt['mu'], scale = paramalt['std'])

    # compute cdfs
    cdfc36 = stats.norm.cdf(x, loc = paramc36['mu'], scale = paramc36['std'])
    cdfalt = stats.norm.cdf(x, loc = paramalt['mu'], scale = paramalt['std'])

    # estimate empirical cdf
    ecdf = ECDF(data1)
    ecdfalt = ECDF(data2)

    # transform data
    c36_transformed = nodim.transform_nodim(paramc36, paramalt, data1)
    paramc36_transformed = nodim.estimate_params(c36_transformed)
    pdfc35_transformed = stats.norm.pdf(x, loc = paramc36_transformed['mu'], scale = paramc36_transformed['std'])
    cdfc36_transformed = stats.norm.cdf(x, loc = paramc36_transformed['mu'], scale = paramc36_transformed['std'])
    ecdf_transformed = ECDF(c36_transformed)

    # ks2 test
    kstest_c36 = stats.ks_2samp(c36_transformed, data2)
    return kstest_c36

rois = ['insula', 'caudate', 'accumbens', 'pallidum',
       'putamen', 'thalamus', 'brain-stem', 'hippocampus', 'amygdala',
       'cingulate', 'frontal', 'temporal', 'parietal', 'occipital',
       'parahippocampalgyrus']

# store ks test results for all rois in dataframe
ks_results = {} 
for roi in rois:
    data_c = data[data['tracer']=='C'][roi]
    data_alt = data[data['tracer']=='a'][roi]
    ksstats = run_dimap_ks_test(data_c, data_alt)
    temp_dict = {roi: {'stats': round(ksstats[0],2), 'p': round(ksstats[1],2)}}
    ks_results.update(temp_dict)

# make ks_results a dataframe
df = pd.DataFrame(ks_results)
df = df.transpose()
df.to_excel(Path(__file__).parent.parent / "results" / "dimap_ks_results.xlsx")

# %%
