#%%
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

from src.logsetup import setup_logging
from src import fs_roi_lut
from src.transform import OneHotPD

DATADIR = Path(__file__).parent.parent / "data"
RESULTDIR = Path(__file__).parent.parent / "results"
WORKLISTFILE = DATADIR / "derivatives" / "combined_alt_cimbi_bpnd_fslobes_pyment.xlsx"

exclude = ["cerebellum", "brain-stem", "scanner", "camera", "mr_strength"]
tracer = None
name = "test"
LOGGER = None
ROIFILE = DATADIR / "rois_fs.json"

#%%
def load_data(worklist_file):
    """Load data from worklist file"""
    data = pd.read_excel(worklist_file, index_col="pet_id")
    return data

def select_tracer(data, tracer):
    """Select data for tracer"""
    if tracer is not None:
        data = data[data["tracer"] == tracer]
    return data

def remove_columns(data, exclude):
    """Remove columns from data"""
    for item in exclude:
        # check if column exists
        if item in data.columns:
            data = data.drop(columns=[item])
    return data

def remove_hrrt_altanserin(data):
    """Remove HRRT altanserin data"""
    n_scans = len(data["subject_id"])
    data = data[
        ((data["tracer"] == "a") & (data["camera"] == "GE")) | (data["tracer"] == "C")
    ]
    n_scans_after = len(data["subject_id"])
    LOGGER.info(f"Removed {n_scans - n_scans_after} HRRT altanserin scans")
    return data

def dummify(data, covariates):
    """Dummify data"""
    # one-hot encode categoricals (dummify)
    data = OneHotPD(covariates).fit_transform(data)
    return data

def remove_second_scan(data):
    """Remove second scan of same subject"""
    n_scans = len(data["subject_id"])
    data = data[~data["subject_id"].duplicated(keep="first")]
    n_scans_after = len(data["subject_id"])
    LOGGER.info(f"Removed {n_scans - n_scans_after} follow-up scans")
    return data

def save_data(data, name, resultdir):
    """Save data to file"""
    data.to_excel(resultdir / f"prepared_data_{name}.xlsx")

def _print_demographics(name, n_subjects, n_scans, n_male, n_female, min_age, max_age, mean_age, std_age, n_hrrt, n_ge, n_3t, n_1p5t):
    LOGGER.info(f"{name}:")
    LOGGER.info(f"Subjects: {n_subjects}, Scans: {n_scans}, Male/Female: {n_male}/{n_female}, Age: {min_age:.1f}/{max_age:.1f}, mean (std): {mean_age:.1f} ({std_age:.1f})")
    LOGGER.info(f"Pet: HRRT - {n_hrrt}, GE - {n_ge}, MR: 3T - {n_3t}, 1.5T - {n_1p5t}")
    # logger.info again with round numbers 

def print_demographics(data):
    """Print demographics"""
    # general demographics
    n_subjects = data['subject_id'].nunique()
    n_scans = len(data['subject_id'])
    n_male = data[data['gender'] == 'Male']['subject_id'].count()
    n_female = data[data['gender'] == 'Female']['subject_id'].count()
    min_age = data['chron_age'].min()
    max_age = data['chron_age'].max()
    mean_age = data['chron_age'].mean()
    std_age = data['chron_age'].std()

    n_hrrt = data[data['camera'] == 'HRRT']['subject_id'].count()
    n_ge = data[data['camera'] == 'GE']['subject_id'].count()
    n_3t = data[data['mr_strength'] == '3T']['subject_id'].count()
    n_1p5t = data[data['mr_strength'] == '1.5T']['subject_id'].count()

    # count unique scanners
    scanners = data['scanner'].unique()
    for scanner in scanners:
        n_scanner = data[data['scanner'] == scanner]['subject_id'].count()
        LOGGER.info(f"Scanner {scanner}: {n_scanner} scans")

    # same for altanserin
    n_altsubjects= data[data['tracer'] == 'a']['subject_id'].count()
    n_altscans = data[data['tracer'] == 'a']['subject_id'].count()
    n_altmale = data[(data['tracer'] == 'a') & (data['gender'] == "Male")]['subject_id'].count()
    n_altfemale = data[(data['tracer'] == 'a') & (data['gender'] == "Female")]['subject_id'].count()
    min_altage = data[data['tracer'] == 'a']['chron_age'].min()
    max_altage = data[data['tracer'] == 'a']['chron_age'].max()
    mean_altage = data[data['tracer'] == 'a']['chron_age'].mean()
    std_altage = data[data['tracer'] == 'a']['chron_age'].std()

    n_alt3T = data[(data['tracer'] == 'a') & (data['mr_strength'] == '3T')]['subject_id'].count()
    n_alt1p5T = data[(data['tracer'] == 'a') & (data['mr_strength'] == '1.5T')]['subject_id'].count()


    # same for c36
    n_c36subjetcs = data[data['tracer'] == 'C']['subject_id'].count()
    n_c36scans = data[data['tracer'] == 'C']['subject_id'].count()
    n_c36male = data[(data['tracer'] == 'C') & (data['gender'] == "Male")]['subject_id'].count()
    n_c36female = data[(data['tracer'] == 'C') & (data['gender'] == "Female")]['subject_id'].count()
    min_c36age = data[data['tracer'] == 'C']['chron_age'].min()
    max_c36age = data[data['tracer'] == 'C']['chron_age'].max()
    mean_c36age = data[data['tracer'] == 'C']['chron_age'].mean()
    std_c36age = data[data['tracer'] == 'C']['chron_age'].std()

    n_c363T = data[(data['tracer'] == 'C') & (data['mr_strength'] == '3T')]['subject_id'].count()
    n_c361p5T = data[(data['tracer'] == 'C') & (data['mr_strength'] == '1.5T')]['subject_id'].count()

    _print_demographics("Demographics Total", n_subjects, n_scans, n_male, n_female, min_age, max_age, mean_age, std_age, n_hrrt, n_ge, n_3t, n_1p5t)
    _print_demographics("Demographics C36", n_c36subjetcs, n_c36scans, n_c36male, n_c36female, min_c36age, max_c36age, mean_c36age, std_c36age, 0, 0, n_c363T, n_c361p5T)
    _print_demographics("Demographics Altanserin", n_altsubjects, n_altscans, n_altmale, n_altfemale, min_altage, max_altage, mean_altage, std_altage, 0, 0, n_alt3T, n_alt1p5T)

def select_rois(data, roifile, covariates):
    """Select rois"""
    LOGGER.info("Reading ROIs from %s", roifile)
    rois = fs_roi_lut.read_roi_names(roifile)
    rois.extend(covariates)

    LOGGER.info("Selecting ROIs %s", rois)

    # append rois with covariates

    # if roi is in data, keep it
    data = data[data.columns[data.columns.isin(rois)]]

    return data

def prepare_data(datafile, roifile, exclude, tracer, name, resultdir, logger):
    """Prepare data for training"""
    global LOGGER
    LOGGER = logger
    data = load_data(datafile)
    data = select_tracer(data, tracer)
    data = remove_hrrt_altanserin(data)
    data = remove_second_scan(data)
    print_demographics(data)
    data = remove_columns(data, exclude)
    data = select_rois(data, roifile, ['tracer', 'pyment', 'chron_age'])
    data = dummify(data,covariates=['tracer'])

    save_data(data, name, resultdir)

# create argparser
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
        "--name", type=str, help="Name of the experiment", default="test"
    )
    parser.add_argument(
        "--roi_file", type=str, help="Path to roi file", default=ROIFILE
    )
    parser.add_argument("--exclude", help="Columns to exclude", nargs="+", default=[])

    parser.add_argument("--tracer", type=str, help="tracer to use (C, a)", default=None)

    return parser

def get_commandline_args():
    parser = get_parser()
    args = parser.parse_args()
    return args

def main():
    args = get_commandline_args()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    worklist_file = data_dir / args.worklist_file
    exclude = args.exclude
    tracer = args.tracer
    name = args.name
    roi_file = args.roi_file

    
    log_name = (
        f"pipe_prepdata_{name}" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    )
    log_file = RESULTDIR / "logs" / log_name
    logger = setup_logging(log_file)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Worklist file: {worklist_file}")
    logger.info(f"Exclude: {exclude}")
    logger.info(f"Tracer: {tracer}")
    logger.info(f"Name: {name}")

    prepare_data(worklist_file, roi_file, exclude, tracer, name, results_dir, logger)

if __name__ == "__main__":
    main()
