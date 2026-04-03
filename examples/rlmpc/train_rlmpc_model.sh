#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D_attitude'

# TASK='stab'
TASK='track'

ALGO='ppo_mpc'
# ALGO='appo_mpc'
# ALGO='sac_mpc'

EXP_NAME='quad_2d_track'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
for SEED in {0..4}
do
    echo "Training ${ALGO} on ${SYS_NAME} with seed ${SEED}"
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./Results/${EXP_NAME} \
        --tag ${SYS}_${ALGO}_03 \
        --seed ${SEED} \
        --kv_overrides \
            task_config.randomized_init=True
        # --use_gpu
done
