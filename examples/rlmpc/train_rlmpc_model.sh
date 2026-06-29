#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_2D_attitude'
SYS='quadrotor_3D_attitude'

# TASK='stab'
TASK='track'

# ALGO='ppo_mpc'
ALGO='appo_mpc'
# ALGO='sac_mpc'

EXP_NAME='test'
TAG="${SYS}_${ALGO}_11"

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
for SEED in {0..0}
do
    echo "Training ${ALGO} on ${SYS_NAME} with seed ${SEED}"
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}_11.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}_11.yaml \
        --output_dir ./Results/${EXP_NAME} \
        --tag ${TAG} \
        --seed ${SEED} \
        --use_gpu \
        --kv_overrides \
            task_config.randomized_init=True
done
wait
# tb-reducer ./Results/${EXP_NAME}/${TAG}/seed* -o ./Results/${EXP_NAME}/${TAG}/ -r mean
