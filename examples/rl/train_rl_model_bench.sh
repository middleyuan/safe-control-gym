#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'
SYS='quadrotor_3D_attitude'

# TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='dppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

EXP_NAME='final'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# TRAIN_LIST=('nominal' 'generalization' 'robustness_dr')
# TRAIN_LIST=('robustness_ob5' 'robustness_ps3' 'robustness_combo')
TRAIN_LIST=('nominal' 'generalization' 'robustness_dr' 'robustness_ob5' 'robustness_ps3' 'robustness_combo')
# shellcheck disable=SC2054
Q=(3.0,0.1,3.0,0.1,0.1,0.001)

NS=1
T=11
H=0
EVAL_LIST=('performance' 'robustness_ob' 'robustness_ps' 'robustness_pm' 'robustness_dw' 'generalization')

# Train the unsafe controller/agent.
for TRAIN in "${TRAIN_LIST[@]}"; do
    if [ "${TRAIN}" == 'nominal' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}.yaml"
    elif [ "${TRAIN}" == 'generalization' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_gen.yaml"
    elif [ "${TRAIN}" == 'robustness_ob5' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob5.yaml"
    elif [ "${TRAIN}" == 'robustness_ps3' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ps3.yaml"
    elif [ "${TRAIN}" == 'robustness_ob5ps3' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob5_ps3.yaml"
    elif [ "${TRAIN}" == 'robustness_combo' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_combo.yaml"
    elif [ "${TRAIN}" == 'robustness_pm' ]; then
      CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
      CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_pm.yaml"
    fi
    echo ${CONFIG1}
    echo ${CONFIG2}

    for SEED in {0..4}; do
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
    done
    # wait

    # RL Experiment
    for EVAL in "${EVAL_LIST[@]}"; do
        if [ "${EVAL}" == 'robustness_ob' ]; then
            EXTERNAL_PARAM=(0 1 2 3 4 5 10 15 20 25 30 40 50 60 70 80 90 100)
        elif [ "${EVAL}" == 'robustness_ps' ]; then
            EXTERNAL_PARAM=(0 1 2 3 4 5 10 15 20 25 30 40 50 60 70 80 90 100)
        elif [ "${EVAL}" == 'robustness_pm' ]; then
            EXTERNAL_PARAM=(0 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)
        elif [ "${EVAL}" == 'robustness_dw' ]; then
            EXTERNAL_PARAM=(1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.5 4.0 4.5 5.0)
        elif [ "${EVAL}" == 'generalization' ]; then
            EXTERNAL_PARAM=(9 10 11 12 13 14 15)
        else
            EXTERNAL_PARAM=(1)
        fi

        for EP in "${EXTERNAL_PARAM[@]}"; do
            for SEED in {0..4}; do
                python3 ./rl_experiment.py \
                    --task ${SYS_NAME} \
                    --algo ${ALGO} \
                    --use_gpu \
                    --overrides \
                        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
                    --experiment_type ${EVAL} \
                    --seed ${SEED} \
                    --kv_overrides \
                        algo_config.training=False \
                        task_config.normalized_rl_action_space=False \
                        task_config.randomized_init=True \
                        task_config.task_info.num_cycles=2 \
                        task_config.episode_len_sec=${T} \
                        task_config.noise_scale=${NS} \
                        task_config.downwash_height=${H} \
                        task_config.external_param=${EP} \
                    --pretrain_path ./Results/${EXP_NAME}/${TRAIN}/${SYS}_${ALGO}_data/seed${SEED}_*/ &
            done
            wait
        done
    done
done
