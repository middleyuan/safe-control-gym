#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'

# TASK='stab'
TASK='track'

# ALGO='q_mpc'
# ALGO='td3_mpc'
ALGO='ppo_mpc'
# ALGO='ppo_vmpc'
# ALGO='appo_mpc'
# ALGO='mpc_lstdq'

EXP_NAME='ppo_mpc'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
for SEED in {0..0}
do
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./Results/${EXP_NAME} \
        --tag ${SYS}_${ALGO}_data \
        --seed ${SEED} \
        --kv_overrides \
            task_config.randomized_init=True
done
