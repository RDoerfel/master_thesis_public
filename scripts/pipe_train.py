#%%
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse
from sklearn import set_config
from sklearn.preprocessing import StandardScaler

from src.trainer import Trainer
from src.logsetup import setup_logging
from src import prep
from src import construct
from src import pipeconfig
from src import fs_roi_lut

# Gobal defaults
DATADIR = Path(__file__).parent.parent / "data"
RESULTDIR = Path(__file__).parent.parent / "results"
WORKLISTFILE = Path("derivatives") / "combined_alt_cimbi_bpnd_fslobes_pyment.xlsx"
#WORKLISTFILE = RESULTDIR / "prepared_data_tracer.xlsx"
PIPELINEFILE = Path("pipe_config.json")
ROIFILE = Path("rois_fs.json")

# functions
def run_pipeline(
    worklist_file,
    pipeline_file,
    pipe_name,
    results_dir,
    roi_file,
    use_mr_brain_age,
    suffix,
    logger,
):
    """Run pipeline for age prediction"""
    # log start
    logger.info("Starting age prediction")

    # load data
    logger.info("Loading data from %s", worklist_file)
    data = pd.read_excel(worklist_file, index_col="pet_id")

    # bin data into age bins for age strata
    data = prep.bin_data(data, "chron_age", min_val=0.0, max_val=100.0, step=10)

    # read rois
    logger.info("Reading ROIs from %s", roi_file)
    rois = fs_roi_lut.read_roi_names(roi_file)

    # exclude cerebellum and brainstem
    # TODO do somewehre else
    rois = [item for item in rois if not "cerebellum" in item]
    rois = [item for item in rois if not "brain-stem" in item]

    logger.info("Using ROIs: %s", rois)

    # get covariates to use
    covariates = pipeconfig.get_covarities(pipeline_file)
    logger.info("Using covarities: %s", covariates)

    # remove brainage from covariates if not used
    numerics = ["pyment"]
    if not use_mr_brain_age:
        data.drop(columns=["pyment"], inplace=True)
        numerics = []

    # create dict to replace strings in config file
    transformer = [("scaler", StandardScaler(), rois + numerics)]
    params_dict = pipeconfig.make_eval_dict(
        steps=["dimap", "scaler"],
        kwargs=["columns", "transformers"],
        values=[rois, transformer],
    )

    # read pipeline copnfiguration
    pipe = pipeconfig.build_pipeline(
        pipeline_file, custom=pipe_name, eval_dict=params_dict
    )

    # get cross-validation splits
    cv_inner, cv_outer = pipeconfig.build_cross_val(pipeline_file)

    # get paramgrid for gridsearch
    param_grid = pipeconfig.get_param_grid(pipeline_file, pipe_name)

    logger.info("Using Cols: %s", data.columns)

    # train
    trainer = Trainer(logger)

    trainer.train(
        data=data,
        strat_col="chron_age_group",
        label_col="chron_age",
        pipeline=pipe,
        cv_inner=cv_inner,
        cv_outer=cv_outer,
        param_grid=param_grid,
    )

    # apply name with pyment if mr brain age is used
    if use_mr_brain_age:
        pipe_name =  "pyment_" + pipe_name

    # get scores
    if suffix:
        suffix = "_" + suffix

    scores = trainer.get_scores()
    scores.to_csv(results_dir / f"pred_age_scores_{pipe_name}{suffix}.csv", index=False)

    # get predictions
    results = trainer.get_results()
    results.to_csv(results_dir / f"pred_age_results_{pipe_name}{suffix}.csv", index=False)

    # get best params
    best_params = trainer.get_best_params()
    best_params.to_csv(results_dir / f"pred_age_best_params_{pipe_name}{suffix}.csv", index=False)

    # get weights
    weights = trainer.get_coefficients()
    if weights is not None:
        weights.to_csv(results_dir / f"pred_age_weights_{pipe_name}{suffix}.csv", index=False)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate age prediction from brain binding potentials MR predicted age"
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATADIR, help="Path to data directory"
    )
    parser.add_argument(
        "--results_dir", type=str, help="Path to results directory", default=RESULTDIR
    )
    parser.add_argument(
        "--worklist_file", type=str, help="Path to worklist file", default=WORKLISTFILE
    )
    parser.add_argument(
        "--pipeline_file",
        type=str,
        help="Path to pipeline config file",
        default=PIPELINEFILE,
    )
    parser.add_argument(
        "--pipe_name", type=str, help="Name of pipeline to use", default="rvm"
    )
    parser.add_argument("--roi_file", type=str, help="ROIs to use", default=ROIFILE)
    parser.add_argument("--mr_brain_age", action='store_true', help="Use MR brain age in predictions as covariate")
    parser.add_argument("--suffix", type=str, help="Note to add to the saved result names", default="")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    worklist_file = data_dir / args.worklist_file
    pipeline_file = data_dir / args.pipeline_file
    roi_file = data_dir / args.roi_file
    pipe_name = args.pipe_name
    use_mr_brain_age = args.mr_brain_age
    suffix = args.suffix

    # preconfigure pipeline display
    set_config(display="diagram")

    # prepare logging
    log_name = (
        f"pred_age_{pipe_name}" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    )
    log_file = results_dir / "logs" / log_name
    logger = setup_logging(log_file)

    # print cli command to reproduce run
    cli_command = "python " + str(Path(__file__))
    for arg in vars(args):
        cli_command = cli_command + f" --{arg} {getattr(args, arg)}"
    logger.info("Running: \n %s", cli_command)

    # run pipeline
    run_pipeline(
        worklist_file,
        pipeline_file,
        pipe_name,
        results_dir,
        roi_file,
        True,
        suffix,
        logger,
    )


if __name__ == "__main__":
    main()
