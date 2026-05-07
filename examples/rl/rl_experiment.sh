#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'
SYS='quadrotor_3D_attitude'

#TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='dppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi
EXP_DATA='test'
SEED=3
SUBSEED=3
EVAL='performance'
# RL Experiment
python3 ./rl_experiment.py \
            --task ${SYS_NAME} \
            --algo ${ALGO} \
            --overrides \
                ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            --seed ${SEED} \
            --use_gpu \
            --experiment_type ${EVAL} \
            --kv_overrides \
                algo_config.training=False \
                task_config.normalized_rl_action_space=False \
                task_config.randomized_init=True \
                task_config.episode_len_sec=${EPISODE_LEN} \
            --pretrain_path /home/benchmark/safe-control-gym/examples/rl/models/${ALGO}/
            # task_config.task_info.custom_snap_ref_traj=${TRAJ_FILE} \