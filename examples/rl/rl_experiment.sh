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

# ENV parameters for each algorithm
if [ "${ALGO}" == 'ppo' ]; then
    # PPO env parameters
    # shellcheck disable=SC2054
    Q=(4.0,0.1,4.0,0.1,4.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
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

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi
EXP_DATA='test'
SEED=1
SUBSEED=11
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
                task_config.randomized_init=True \
                task_config.episode_len_sec=${EPISODE_LEN} \
                task_config.task_info.custom_snap_ref_traj=${TRAJ_FILE} \
                task_config.rew_state_weight=${Q} \
            --pretrain_path ./Results/${EXP_DATA}/${SYS}_${ALGO}_data_obstacle_seed${SEED}/seed${SUBSEED}_*/