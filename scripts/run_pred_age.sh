# run age prediction pipeline for different models
# Usage: bash run_pred_age.sh

# setup paths
DATADIR=/data1/RDoerfel/master_thesis/data
RESULTDIR=/data1/RDoerfel/master_thesis/results
WORKLISTFILE=derivatives/combined_alt_cimbi_bpnd_fslobes_pyment.xlsx
PIPECONFIGFILE=pipe_config.json
ROIFILE=rois_fs.json
# define pipeline names
PIPES=(bayesridge rvm gpr)



# run pipeline for each model
for PIPE in ${PIPES[@]}; do
    echo "Running pipeline for model: ${PIPE}"
    # run pipeline
    python /data1/RDoerfel/master_thesis/scripts/pipe_train.py \
        --data_dir $DATADIR \
        --results_dir $RESULTDIR  \
        --worklist_file $WORKLISTFILE \
        --pipeline_file $PIPECONFIGFILE \
        --pipe_name $PIPE \
        --roi_file $ROIFILE \
        --note tracer \
        --mr_brain_age
    
        python /data1/RDoerfel/master_thesis/scripts/pipe_train.py \
        --data_dir $DATADIR \
        --results_dir $RESULTDIR  \
        --worklist_file $WORKLISTFILE \
        --pipeline_file $PIPECONFIGFILE \
        --pipe_name $PIPE \
        --roi_file $ROIFILE \
        --note tracer 
done
