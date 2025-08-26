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

# ENV parameters for each algorithm
if [ "${ALGO}" == 'ppo' ]; then
    # PPO env parameters
    # shellcheck disable=SC2054
    Q=(5.0,0.1,5.0,0.1,5.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    # Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
elif [ "${ALGO}" == 'dppo' ]; then
    # DPPO env parameters
    # shellcheck disable=SC2054
    Q=(5.0,0.1,5.0,0.1,5.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
elif [ "${ALGO}" == 'sac' ]; then
    # SAC env parameters
    # shellcheck disable=SC2054
    Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
fi

EXP_NAME='test'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Removed the temporary data used to train the new unsafe model.
# rm -r -f ./${ALGO}_data/

if [ "$ALGO" == 'safe_explorer_ppo' ]; then
    # Pretrain the unsafe controller/agent.
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}_pretrain.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./unsafe_rl_temp_data/ \
        --seed 2 \
        --kv_overrides \
            task_config.init_state=None

    # Move the newly trained unsafe model.
    mv ./unsafe_rl_temp_data/model_latest.pt ./models/${ALGO}/${ALGO}_pretrain_${SYS}_${TASK}.pt

    # Removed the temporary data used to train the new unsafe model.
    rm -r -f ./unsafe_rl_temp_data/
fi

# Train the unsafe controller/agent.
# SEEDS=(1 2 3)
SEEDS=(1 2 3)
# Loop through each SEED
for SEED in "${SEEDS[@]}"; do
    echo "Running with SEED: $SEED"
    
    # Set episode_len_sec and trajectory file based on SEED
    if [ "$SEED" -eq 1 ]; then
        EPISODE_LEN=23.24
        TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj_final_1_vf_f.npy"
    elif [ "$SEED" -eq 2 ]; then
        EPISODE_LEN=29.05
        TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj_final_2_vf_f.npy"
    elif [ "$SEED" -eq 3 ]; then
        EPISODE_LEN=43.575
        TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj_final_3_vf_f.npy"
    elif [ "$SEED" -eq 0 ]; then
        EPISODE_LEN=25.259999999999998
        TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj.npy"
    else
        EPISODE_LEN=30  # Default value
        TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj_final_1_vf.npy"  # Default trajectory
    fi
    
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./Results/${EXP_NAME} \
        --tag ${SYS}_${ALGO}_data_obstacle_seed${SEED} \
        --seed ${SEED} \
        --use_gpu \
        --kv_overrides \
            task_config.randomized_init=True \
            task_config.normalized_rl_action_space=False \
            task_config.episode_len_sec=${EPISODE_LEN} \
            task_config.task_info.custom_snap_ref_traj=${TRAJ_FILE} \
            task_config.rew_state_weight=${Q}
done

# Move the newly trained unsafe model.
# mv ./unsafe_rl_temp_data/model_best.pt ./models/${ALGO}/${ALGO}_model_${SYS}_${TASK}.pt

# Removed the temporary data used to train the new unsafe model.
# rm -r -f ./unsafe_rl_temp_data/
