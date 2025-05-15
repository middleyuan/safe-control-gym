#!/bin/bash

cd ~/safe-control-gym

localOrHost=$1
sys=$2 # cartpole, or quadrotor_2D_attitude
sys_name=${sys%%_*} # cartpole, or quadrotor
algo=$3
prior=$4
safety_filter=$5 # True or False
task=$6 # stab, or tracking
FOLDER="./examples/hpo/hpo/${algo}"
OUTPUT_DIR=(${FOLDER})
hpo_postfix=$7 # ""  "_eval" "_basic" "_dw_h=1dot5" "_dw_h=2dot5" "_dw_h=4" "_ob_ns=5" "_ob_ns=15" "_ob_ns=25" "_proc_ns=5" "_proc_ns=15" "_proc_ns=25"

# activate the environment
if [ "$localOrHost" == 'local' ]; then
    source /home/tsung/anaconda3/etc/profile.d/conda.sh
    conda activate safe
elif [ "$localOrHost" == 'host0' ]; then
    source /home/tueilsy-st01/anaconda3/etc/profile.d/conda.sh
    conda activate safe
elif [ "$localOrHost" == 'hostx' ]; then
    source /home/tsung/miniconda3/etc/profile.d/conda.sh
    conda activate safe
elif [ "$localOrHost" == 'cluster' ]; then
    echo "Doing experiment in cluster..."
else
    echo "Please specify the machine to run the experiment."
    exit 1
fi

# echo config path
echo "task config path: ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}${hpo_postfix}.yaml"
echo "algo config path: ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml"
echo "hpo config path: ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml"

    # Run the process in the background
    python ./examples/hpo/hpo_experiment.py \
        --algo "${algo}" \
        --task "${sys_name}" \
        --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}${hpo_postfix}.yaml \
                    ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                    ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml \
        --output_dir "${OUTPUT_DIR}" \
        --func eval \
        --use_gpu True &