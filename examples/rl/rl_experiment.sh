#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'
# SYS='quadrotor_3D_attitude'
if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi
#TASK='stab'
TASK='track'

# ALGO='ppo'
# ALGO='dppo'
ALGO='sac'
# ALGO='safe_explorer_ppo'

EXP_DATA='test'
SEED=0
SUBSEED=14
EVAL='performance'

# RL Experiment
python3 ./rl_experiment.py \
            --task ${SYS_NAME} \
            --algo ${ALGO} \
            --use_gpu \
            --overrides \
                ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            --seed ${SEED} \
            --experiment_type ${EVAL} \
            --kv_overrides \
                algo_config.training=False \
                task_config.normalized_rl_action_space=False \
                task_config.randomized_init=True
            # --pretrain_path ./Results/${EXP_DATA}/${SYS}_${ALGO}_data_seed${SEED}/seed${SUBSEED}_*/