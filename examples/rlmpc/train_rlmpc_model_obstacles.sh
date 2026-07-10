#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_2D_attitude'
SYS='quadrotor_3D_attitude'

# TASK='stab'
TASK='track'

ALGO='ppo_mpc'
# ALGO='appo_mpc'
# ALGO='sac_mpc'

EXP_NAME='quad_3d'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
for SEED in {0..0}
do
    for LEN in {13,15,20}
    do
        echo "Training ${ALGO} on ${SYS_NAME} for data length ${LEN} with seed ${SEED}"
        TAG="obstacle_course/${SYS}_${ALGO}_${LEN}"
        python3 ../../safe_control_gym/experiments/train_rl_controller.py \
            --algo ${ALGO} \
            --task ${SYS_NAME} \
            --overrides \
                ./config_overrides/${SYS}/${ALGO}_${SYS}_${LEN}.yaml \
                ./config_overrides/${SYS}/obstacle_course_${LEN}.yaml \
            --output_dir ./Results/${EXP_NAME} \
            --tag ${TAG} \
            --seed ${SEED} \
            --kv_overrides \
                task_config.randomized_init=True
    done
    wait
    # tb-reducer ./Results/${EXP_NAME}/${TAG}/seed* -o ./Results/${EXP_NAME}/${TAG}/ -r mean
done
