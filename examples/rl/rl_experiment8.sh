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

    # Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.5)
    # Q=(4.0,0.1,4.0,0.1,4.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    # Q=(5.0,0.1,5.0,0.1,5.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)

    R=(0.1,0.1,0.1,0.1)
    # R=(0.25,0.25,0.25,0.25)
    # R=(0.5,0.5,0.5,0.5)
    # R=(1.0,1.0,1.0,1.0)

elif [ "${ALGO}" == 'dppo' ]; then
    # DPPO env parameters
    # shellcheck disable=SC2054
    # Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    # Q=(4.0,0.1,4.0,0.1,4.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    Q=(5.0,0.1,5.0,0.1,5.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)

    R=(0.1,0.1,0.1,0.1)
    # R=(0.25,0.25,0.25,0.25)
    # R=(0.5,0.5,0.5,0.5)
    # R=(1.0,1.0,1.0,1.0)

elif [ "${ALGO}" == 'sac' ]; then
    # SAC env parameters
    # shellcheck disable=SC2054
    Q=(3.0,0.1,3.0,0.1,3.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    # Q=(4.0,0.1,4.0,0.1,4.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)
    # Q=(5.0,0.1,5.0,0.1,5.0,0.1,0.1,0.1,0.1,0.001,0.001,0.001,0.001)

    R=(0.1,0.1,0.1,0.1)
    # R=(0.25,0.25,0.25,0.25)
    # R=(0.5,0.5,0.5,0.5)
    # R=(1.0,1.0,1.0,1.0)
fi

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi
EXP_DATA='figure8'
SEED=30
SUBSEED=30
# Set episode_len_sec and trajectory file based on SEED
if [ "$SEED" -eq 7 ]; then
    EPISODE_LEN=7
elif [ "$SEED" -eq 11 ]; then
    EPISODE_LEN=11
elif [ "$SEED" -eq 30 ]; then
    EPISODE_LEN=30
elif [ "$SEED" -eq 3 ]; then
    EPISODE_LEN=15.5
else    
    EPISODE_LEN=20.5  # Default value
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
                task_config.rew_state_weight=${Q} \
            --pretrain_path /home/benchmark/safe-control-gym/examples/rl/models/ppo/
            # --pretrain_path ./Results/${EXP_DATA}/${SYS}_${ALGO}_data_seed${SEED}/seed${SUBSEED}_*/
            # --pretrain_path ./Results/${EXP_DATA}/${SYS}_${ALGO}_data_obstacle_seed${SEED}/seed${SUBSEED}_*/
            # --pretrain_path /home/benchmark/safe-control-gym/examples/rl/models/ppo/
            # task_config.task_info.custom_snap_ref_traj=${TRAJ_FILE} \