#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'
SYS='quadrotor_3D_attitude'

# TASK='stab'
TASK='track'

# List of algorithms to train
ALGOS=('ppo' 'dppo' 'sac')

EXP_NAME='obstacle'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
SEEDS=(3)

# Loop through each algorithm
for ALGO in "${ALGOS[@]}"; do
    echo "======================================"
    echo "Training algorithm: $ALGO"
    echo "======================================"

    # ENV parameters for each algorithm
    if [ "${ALGO}" == 'ppo' ]; then
    # PPO env parameters
    # shellcheck disable=SC2054

    # Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    # Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    # Q=(4.0,0.1,4.0,0.1,4.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    Q=(5.0,0.1,5.0,0.1,5.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)

    # R=(0.1,0.1,0.1,0.1)
    R=(0.25,0.25,0.25,0.25)
    # R=(0.1,0.1,0.1,0.1)
    # R=(1.0,1.0,1.0,1.0)

    elif [ "${ALGO}" == 'dppo' ]; then
        # DPPO env parameters
        # shellcheck disable=SC2054
        # Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
        # Q=(4.0,0.1,4.0,0.1,4.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
        Q=(5.0,0.1,5.0,0.1,5.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)

        # R=(0.1,0.1,0.1,0.1)
        R=(0.15,0.15,0.15,0.15)
        # R=(0.5,0.5,0.5,0.5)
        # R=(1.0,1.0,1.0,1.0)

    elif [ "${ALGO}" == 'sac' ]; then
        # SAC env parameters
        # shellcheck disable=SC2054
        # Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
        # Q=(4.0,0.1,4.0,0.1,4.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
        Q=(5.0,0.1,5.0,0.1,5.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)

        # R=(0.1,0.1,0.1,0.1)
        # R=(0.25,0.25,0.25,0.25)
        R=(0.5,0.5,0.5,0.5)
        # R=(1.0,1.0,1.0,1.0)
    fi

    # Loop through each SEED
    for SEED in "${SEEDS[@]}"; do
        echo "Running ${ALGO} with SEED: $SEED"

        # Set episode_len_sec and trajectory file based on SEED
        if [ "$SEED" -eq 0 ]; then
            EPISODE_LEN=13.0
            TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj_final_v6_0.npy"
        elif [ "$SEED" -eq 1 ]; then
            EPISODE_LEN=15.5
            TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj_final_v6_1.npy"
        elif [ "$SEED" -eq 2 ]; then
            EPISODE_LEN=20.5
            TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj_final_v6_2.npy"
        elif [ "$SEED" -eq 3 ]; then
            EPISODE_LEN=20.5
            TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj.npy"
        else
            EPISODE_LEN=15.5  # Default value
            TRAJ_FILE="/home/benchmark/safe-control-gym/benchmarking_sim/quadrotor/data/custom_snap_ref_traj.npy"  # Default trajectory
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
                task_config.rew_state_weight=${Q} \
                task_config.rew_act_weight=${R}
    done

    echo "Completed training for ${ALGO}"
    echo ""
done

echo "All algorithms training completed!"