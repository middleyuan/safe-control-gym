#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'
# SYS='quadrotor_3D_attitude'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='dppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

EXP_NAME='Final'
TRAIN_LIST=('nominal' 'generalization' 'robustness_pm' 'robustness_ob5' 'robustness_ps3' 'robustness_combo')
EVAL_LIST=('performance' 'generalization' 'robustness_ob' 'robustness_ps' 'robustness_pm')

# ENV parameters for each algorithm
if [ "${ALGO}" == 'ppo' ]; then
    # PPO env parameters
    # shellcheck disable=SC2054
    Q=(5.0,0.1,5.0,0.1,0.1,0.001)
elif [ "${ALGO}" == 'dppo' ]; then
    # DPPO env parameters
    # shellcheck disable=SC2054
    Q=(5.0,0.1,5.0,0.1,0.1,0.001)
elif [ "${ALGO}" == 'sac' ]; then
    # SAC env parameters
    # shellcheck disable=SC2054
    Q=(10.0,0.1,10.0,0.1,0.1,0.001)
fi

# Train the unsafe controller/agent.
for TRAIN in "${TRAIN_LIST[@]}"; do
    if [ "${TRAIN}" == 'nominal' ]; then
        CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
        CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}.yaml"
    elif [ "${TRAIN}" == 'traj_data' ]; then
        CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
        CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}.yaml"
    elif [ "${TRAIN}" == 'generalization' ]; then
        CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
        CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_gen.yaml"
    elif [ "${TRAIN}" == 'robustness_ob5' ]; then
        CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}_dr.yaml"
        CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob5.yaml"
    elif [ "${TRAIN}" == 'robustness_ps3' ]; then
        CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}_dr.yaml"
        CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ps3.yaml"
    elif [ "${TRAIN}" == 'robustness_ob5ps3' ]; then
        CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}_dr.yaml"
        CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob5ps3.yaml"
    elif [ "${TRAIN}" == 'robustness_pm' ]; then
        # shellcheck disable=SC2054
        Q=(1.0,0.1,1.0,0.1,0.1,0.001)
        CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}_dr.yaml"
        CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_pm.yaml"
    elif [ "${TRAIN}" == 'robustness_combo' ]; then
        # shellcheck disable=SC2054
        Q=(1.0,0.1,1.0,0.1,0.1,0.001)
        CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}_dr.yaml"
        CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_combo.yaml"
    elif [ "${TRAIN}" == 'robustness_dw' ]; then
        CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}_dr.yaml"
        CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_dw.yaml"
    fi
    echo ${CONFIG1}
    echo ${CONFIG2}

    for SEED in {0..9}; do
        python3 ../../safe_control_gym/experiments/train_rl_controller.py \
            --algo ${ALGO} \
            --task ${SYS_NAME} \
            --overrides \
                "${CONFIG1}" \
                "${CONFIG2}" \
            --output_dir ./Results/${EXP_NAME}/${TRAIN} \
            --tag ${SYS}_${ALGO}_data \
            --seed "${SEED}" \
            --use_gpu \
            --kv_overrides \
                task_config.randomized_init=True \
                task_config.normalized_rl_action_space=False\
                task_config.rew_state_weight=${Q}

        # Evaluate the trained controller/agent.
        # RL Experiment
        wait
        for EVAL in "${EVAL_LIST[@]}"; do
            echo "Evaluating ${EVAL} for ${SYS}_${ALGO} with seed ${SEED}"
            if [ "${EVAL}" == 'robustness_ob' ]; then
                EXTERNAL_PARAM=(0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
            elif [ "${EVAL}" == 'robustness_ps' ]; then
                EXTERNAL_PARAM=(0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
            elif [ "${EVAL}" == 'robustness_pm' ]; then
                EXTERNAL_PARAM=(0 0.05 0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0)
            elif [ "${EVAL}" == 'robustness_dw' ]; then
                EXTERNAL_PARAM=(1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.5 4.0 4.5 5.0)
            elif [ "${EVAL}" == 'generalization' ]; then
                EXTERNAL_PARAM=(9 10 11 12 13 14 15)
            else
                EXTERNAL_PARAM=(1)
            fi

            for EP in "${EXTERNAL_PARAM[@]}"; do
                python3 ./rl_experiment.py \
                    --task ${SYS_NAME} \
                    --algo ${ALGO} \
                    --use_gpu \
                    --overrides \
                        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                        "${CONFIG1}" \
                    --experiment_type ${EVAL} \
                    --seed ${SEED} \
                    --kv_overrides \
                        algo_config.training=False \
                        task_config.normalized_rl_action_space=False \
                        task_config.randomized_init=True \
                        task_config.external_param=${EP} \
                    --pretrain_path ./Results/${EXP_NAME}/${TRAIN}/${SYS}_${ALGO}_data/seed${SEED}_*/
            done
            wait
        done
    done
    wait
done
