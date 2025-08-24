#!/bin/bash

#SYS='cartpole'
#SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
SYS='quadrotor_3D_attitude'

#TASK='stab'
TASK='track'

ALGO='ppo_mpc'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

python3 ./rlmpc_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --seed 0 \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
    --kv_overrides \
        algo_config.training=False \
        task_config.randomized_init=False
