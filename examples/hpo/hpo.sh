#!/bin/bash

######## NOTE ########
# This script is used to run HPO in parallel.
# 1. Adjust hpo config.
# 2. Remove or backup the database if needed.
# 3. Create a screen session screen, and detach it Ctrl+a d.
# 4. Run this script by giving experiment name as the first arg, seed as the second, and number of parallel jobs as the third arg.
# 5. If you want to kill them, run pkill -f "python ./.py".
#####################

experiment_name=$1
seed1=$2
parallel_jobs=$3 # Number of parallel jobs
sampler=$4 # optuna or vizier
localOrHost=$5
sys=$6 # cartpole, or quadrotor_2D_attitude
sys_name=${sys%%_*} # cartpole, or quadrotor
algo=$7 # ilqr, gpmpc_acados, ppo
prior=$8
safety_filter=$9 # True or False
task=${10} # stab, or tracking
resume=${11} # 0 or 1
hpo_postfix=${12} # ""  "_eval" "_basic" "_dw_h=1dot5" "_dw_h=2dot5" "_dw_h=4" "_ob_ns=5" "_ob_ns=15" "_ob_ns=25" "_proc_ns=5" "_proc_ns=15" "_proc_ns=25"

# activate the environment
if [ "$localOrHost" == 'local' ]; then
    # source /home/tsung/anaconda3/etc/profile.d/conda.sh
    # ~/miniconda3/etc/profile.d/conda.sh
    # conda activate safe
    # cd ~/safe-control-gym
    cd ~/LSY_Projects/safe_control_gym/safe-control-gym
elif [ "$localOrHost" == 'host0' ]; then
    source /home/tueilsy-st01/anaconda3/etc/profile.d/conda.sh
    conda activate safe
    cd ~/safe-control-gym
elif [ "$localOrHost" == 'hostx' ]; then
    source /home/tsung/miniconda3/etc/profile.d/conda.sh
    conda activate safe
    cd ~/safe-control-gym
elif [ "$localOrHost" == 'cluster' ]; then
    echo "Doing experiment in cluster..."
else
    echo "Please specify the machine to run the experiment."
    exit 1
fi

# Adjust the seed for each parallel job
seeds=()
for ((i=0; i<parallel_jobs; i++)); do
    seeds[$i]=$((seed1 + i * 100))
done

# if resume is 1 and sampler is optuna, load the study for all jobs
if [ "$resume" == '1' ] && [ "$sampler" == 'optuna' ]; then

    if [ "$safety_filter" == 'False' ]; then
        algo_name=${algo}
        echo "task config path: ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}${hpo_postfix}.yaml"
        echo "algo config path: ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}_${prior}.yaml"
        echo "hpo config path: ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_hpo.yaml"
        for ((i=0; i<parallel_jobs; i++)); do
            python ./examples/hpo/hpo_experiment.py \
                                --algo $algo \
                                --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}${hpo_postfix}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}_${prior}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_hpo.yaml \
                                --output_dir ./examples/hpo/hpo/${algo_name} \
                                --sampler $sampler \
                                --resume ${resume} \
                                --use_gpu True \
                                --task ${sys_name} --load_study True --tag ${experiment_name} --seed ${seeds[$i]} &
            pids[$i]=$!
            sleep 3
        done
    else
        algo_name=${algo}_mpsc
        echo "task config path: ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}${hpo_postfix}.yaml"
        echo "algo config path: ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}_${prior}.yaml"
        echo "hpo config path: ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_hpo.yaml"
        echo "sf config path: ./examples/hpo/${sys_name}/config_overrides/nl_mpsc_${sys}.yaml"
        for ((i=0; i<parallel_jobs; i++)); do
            python ./examples/hpo/hpo_experiment.py \
                                --algo $algo \
                                --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}${hpo_postfix}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}_${prior}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_hpo.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/nl_mpsc_${sys}.yaml \
                                --kv_overrides sf_config.cost_function=one_step_cost \
                                             sf_config.soften_constraints=True \
                                             algo_config.filter_train_actions=True \
                                             algo_config.penalize_sf_diff=True \
                                             algo_config.sf_penalty=0.03 \
                                --output_dir ./examples/hpo/hpo/${algo_name} \
                                --sampler $sampler \
                                --resume ${resume} \
                                --use_gpu True \
                                --task ${sys_name} --load_study True --tag ${experiment_name} --seed ${seeds[$i]} &
            pids[$i]=$!
            sleep 3
        done
    fi

# else create a study for the first job and load it for the remaining jobs
else
    # First job creates the study
    if [ "$safety_filter" == 'False' ]; then
        algo_name=${algo}
        echo "task config path: ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}${hpo_postfix}.yaml"
        echo "algo config path: ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}_${prior}.yaml"
        echo "hpo config path: ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_hpo.yaml"
        python ./examples/hpo/hpo_experiment.py \
                            --algo $algo \
                            --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}${hpo_postfix}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}_${prior}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_hpo.yaml \
                            --output_dir ./examples/hpo/hpo/${algo_name} \
                            --sampler $sampler \
                            --resume ${resume} \
                            --use_gpu True \
                            --task ${sys_name} --tag ${experiment_name} --seed ${seeds[0]} &
        pid1=$!
        pids[0]=$pid1

        # wait until the first study is created
        sleep 3

        # Remaining jobs load the study
        for ((i=1; i<parallel_jobs; i++)); do
            python ./examples/hpo/hpo_experiment.py \
                                --algo $algo \
                                --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}${hpo_postfix}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml \
                                --output_dir ./examples/hpo/hpo/${algo_name} \
                                --sampler $sampler \
                                --resume ${resume} \
                                --use_gpu True \
                                --task ${sys_name} --load_study True --tag ${experiment_name} --seed ${seeds[$i]} &
            pids[$i]=$!
        done
    fi

    if [ "$safety_filter" == 'True' ]; then
        algo_name=${algo}_mpsc
        echo "task config path: ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}.yaml"
        echo "algo config path: ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}_${prior}.yaml"
        echo "hpo config path: ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_hpo.yaml"
        echo "sf config path: ./examples/hpo/${sys_name}/config_overrides/nl_mpsc_${sys}.yaml"
        python ./examples/hpo/hpo_experiment.py \
                            --algo $algo \
                            --overrides ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}_${prior}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_hpo.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/nl_mpsc_${sys}.yaml \
                            --kv_overrides sf_config.cost_function=one_step_cost \
                                             sf_config.soften_constraints=True \
                                             algo_config.filter_train_actions=True \
                                             algo_config.penalize_sf_diff=True \
                                             algo_config.sf_penalty=0.03 \
                            --output_dir ./examples/hpo/hpo/${algo_name} \
                            --sampler $sampler \
                            --resume ${resume} \
                            --use_gpu True \
                            --task ${sys_name} --tag ${experiment_name} --seed ${seeds[0]} &
        pid1=$!
        pids[0]=$pid1

        # wait until the first study is created
        sleep 3

        for ((i=1; i<parallel_jobs; i++)); do
            python ./examples/hpo/hpo_experiment.py \
                                --algo $algo \
                                --overrides ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_${task}_${prior}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo_name}_${sys}_hpo.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/nl_mpsc_${sys}.yaml \
                                --kv_overrides sf_config.cost_function=one_step_cost \
                                             sf_config.soften_constraints=True \
                                             algo_config.filter_train_actions=True \
                                             algo_config.penalize_sf_diff=True \
                                             algo_config.sf_penalty=0.03 \
                                --output_dir ./examples/hpo/hpo/${algo_name} \
                                --sampler $sampler \
                                --resume ${resume} \
                                --use_gpu True \
                                --task ${sys_name} --load_study True --tag ${experiment_name} --seed ${seeds[$i]} &
            pids[$i]=$!
            sleep 3
        done
    fi
fi

# Wait for all jobs to finish
for pid in ${pids[*]}; do
    wait $pid
    echo "Job $pid finished"
done

# back up the database after all jobs finish
echo "backing up the database"
mv ${algo_name}_hpo${hpo_postfix}_${sampler}.db ./examples/hpo/hpo/${algo_name}/${experiment_name}/${algo_name}_hpo${hpo_postfix}_${sampler}.db
mv ${algo_name}_hpo${hpo_postfix}_${sampler}.db-journal ./examples/hpo/hpo/${algo_name}/${experiment_name}/${algo_name}_hpo${hpo_postfix}_${sampler}.db-journal
mv ${algo_name}_hpo${hpo_postfix}_${sampler}_endpoint.yaml ./examples/hpo/hpo/${algo_name}/${experiment_name}/${algo_name}_hpo${hpo_postfix}_${sampler}_endpoint.yaml
