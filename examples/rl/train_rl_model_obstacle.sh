#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'
SYS='quadrotor_3D_attitude'

# TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='sac'
# ALGO='dppo'
# ALGO='safe_explorer_ppo'

EXP_NAME='test2'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
SEEDS=(11)
# Loop through each SEED
for SEED in "${SEEDS[@]}"; do
    echo "Running with SEED: $SEED"

    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            ./config_overrides/${SYS}/obstacle_course_20.yaml \
        --output_dir ./Results/${EXP_NAME} \
        --tag ${SYS}_${ALGO}_data_obstacle \
        --seed ${SEED} \
        --use_gpu \
        --kv_overrides \
            task_config.randomized_init=True \
            task_config.normalized_rl_action_space=False 
done
