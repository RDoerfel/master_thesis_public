# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
from scipy import stats
import argparse
from sklearn.metrics import mean_absolute_error, r2_score

from src import plotting
from src.statistics import CorrelatedTTest

from src.logsetup import setup_logging


#%% set paths
# path to the results
RESULTDIR = Path(__file__).parent.parent / "results"
DATAFILE = Path(__file__).parent.parent / "data" / "derivatives" / "combined_alt_cimbi_bpnd_fslobes_pyment.xlsx"
PIPELINES = ["bayesridge", "rvm", "gpr"]

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

def run_evaluation(result_dir, pipelines, data_file, suffix, nruns, logger):
    # append pyment_ to pipelines to indicate estimates with pyment as covariate
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

    # append with true, fold and pyment (same for all pipelines)
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
    result_table = pd.DataFrame(index=pipelines)

    score_names = ['mae', 'r2', 'corr', 'corr_pad']
    for score in score_names:
        temp = scores[[f'{pipe}_{score}' for pipe in pipelines]].mean()
        temp.rename(index=lambda s: s.removesuffix(f'_{score}'), inplace=True)
        result_table[score.upper()] = temp.round(2)

    # prepare dataframe for statistical tests
    for pipe in pipelines:
        if pipe == 'pyment':
            continue
        a = scores['pyment_mae'].to_numpy()
        b = scores[f'{pipe}_mae'].to_numpy()
        ttest = CorrelatedTTest(a, b, nruns)
        tstat, p, cil, ciu = ttest.ttest()
        pl, pr = ttest.probabilities()
        result_table.loc[pipe, 'tstat'] = round(tstat,2)
        result_table.loc[pipe, 'p'] = round(p,2)
        result_table.loc[pipe, 'ci'] = f"({cil:.2f}, {ciu:.2f})"
        result_table.loc[pipe, 'plr'] = f"({pl:.2f}, {pr:.2f})"

    # print table
    logger.info(result_table)

    # save table
    result_table.to_excel(result_dir / f'result{suffix}_table.xlsx')


#%%

def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate age prediction from brain binding potentials MR predicted age"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=RESULTDIR,
        help="Path to directory containing results",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        nargs="+",
        default=PIPELINES,
        help="Names of pipelines to evaluate",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default=DATAFILE,
        help="Path to directory containing data to get mr brain age predictions",
    )
    parser.add_argument("--suffix", type=str, default="no_tracer", help="Suffix of data file")
    parser.add_argument("--nruns", type=int, default=20, help="Number of runs")
    return parser


def get_command(args):
    cli_command = "python " + str(Path(__file__))
    for arg in vars(args):
        cli_command = cli_command + f" --{arg} {getattr(args, arg)}"
    return cli_command


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(get_command(args))

    result_dir = Path(args.result_dir)
    data_file = Path(args.data_file)
    pipelines = args.pipelines
    suffix = args.suffix
    nruns = args.nruns

    # prepare logging
    log_name = (
        f"pred_age_{suffix}" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    )
    log_file = RESULTDIR / "logs" / log_name
    logger = setup_logging(log_file)

    run_evaluation(result_dir, pipelines, data_file, suffix, nruns, logger)


if __name__ == "__main__":
    main()
