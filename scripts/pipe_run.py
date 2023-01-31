#%%
from pathlib import Path
import argparse
from src.logsetup import setup_logging
import pipe_prepdata, pipe_performance, pipe_train
from datetime import datetime

RESULTDIR = Path(__file__).parent.parent / "results"
DATADIR= Path(__file__).parent.parent / "data" 
WORKLISTFILE = DATADIR / "derivatives" / "combined_alt_cimbi_bpnd_fslobes_pyment.xlsx"
PIPELINEFILE = DATADIR / "pipe_config.json"
ROIFILE = DATADIR / "rois_fs.json"
pipelines = ["dummy"]
exclude = ["cerebellum", "brain-stem", "scanner", "camera", "mr_strength", "subject_id"]
tracer = None
name = "dummy"

#%%
# prepare logging
log_name = (
    f"pred_age_{name}" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
)
log_file = RESULTDIR / "logs" / log_name
logger = setup_logging(log_file)

#%%
pipe_prepdata.prepare_data(WORKLISTFILE, ROIFILE, exclude, tracer, name, RESULTDIR, logger)
WORKLISTFILE = RESULTDIR / f"prepared_data_{name}.xlsx"
for pipe_name in pipelines:
    pipe_train.run_pipeline(WORKLISTFILE,PIPELINEFILE,pipe_name,RESULTDIR,ROIFILE,True,name,logger)
    pipe_train.run_pipeline(WORKLISTFILE,PIPELINEFILE,pipe_name,RESULTDIR,ROIFILE,False,name,logger)

pipe_performance.run_evaluation(RESULTDIR, pipelines, WORKLISTFILE, name, 20,logger)

#%%
def run(result_dir, name, worklist_file, roifile, exclude, pipelines, tracer, pipeline_file):
    log_name = (
        f"pred_age_{name}" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    )
    log_file = RESULTDIR / "logs" / log_name
    logger = setup_logging(log_file)

    pipe_prepdata.prepare_data(worklist_file, roifile, exclude, tracer, name, result_dir, logger)
    data_file = result_dir / f"prepared_data_{name}.xlsx"
    for pipe_name in pipelines:
        pipe_train.run_pipeline(data_file,pipeline_file,pipe_name,result_dir,roifile,True,name,logger)
        pipe_train.run_pipeline(data_file,pipeline_file,pipe_name,result_dir,roifile,False,name,logger)

    pipe_performance.run_evaluation(result_dir, pipelines, data_file, name, 20, logger)


