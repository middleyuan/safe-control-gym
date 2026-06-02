#!/bin/bash

SYS_NAME='quadrotor'
# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'
# SYS='quadrotor_3D_attitude'

# TASK='stab'
TASK='track'

# ALGO='ppo_mpc'
# ALGO='sac_mpc'
ALGO_LIST=('ppo_mpc')

EXP_NAME='Final_may_3/h_1'
# TRAIN_LIST=('nominal' 'generalization' 'robustness_combo' 'robustness_pm' 'robustness_ob5' 'robustness_ps3')
TRAIN_LIST=('nominal')
EVAL_LIST=('performance' 'generalization' 'robustness_ob' 'robustness_ps' 'robustness_pm')

# Train the unsafe controller/agent.
for ALGO in "${ALGO_LIST[@]}"; do
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
            CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
            CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob5.yaml"
        elif [ "${TRAIN}" == 'robustness_ps3' ]; then
            CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
            CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ps3.yaml"
        elif [ "${TRAIN}" == 'robustness_ob5ps3' ]; then
            CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
            CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_ob5ps3.yaml"
        elif [ "${TRAIN}" == 'robustness_pm' ]; then
            CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}_dr.yaml"
            CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_pm.yaml"
        elif [ "${TRAIN}" == 'robustness_combo' ]; then
            CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}_dr.yaml"
            CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_combo.yaml"
        elif [ "${TRAIN}" == 'robustness_dw' ]; then
            CONFIG1="./config_overrides/${SYS}/${ALGO}_${SYS}.yaml"
            CONFIG2="./config_overrides/${SYS}/${SYS}_${TASK}_dw.yaml"
        fi
        echo ${CONFIG1}
        echo ${CONFIG2}
        TAG="${SYS}_${ALGO}_data2"

        for SEED in {0..4}; do
            python3 ../../safe_control_gym/experiments/train_rl_controller.py \
                --algo ${ALGO} \
                --task ${SYS_NAME} \
                --overrides \
                    "${CONFIG1}" \
                    "${CONFIG2}" \
                --output_dir ./Results/sim/${EXP_NAME}/${TRAIN} \
                --tag ${TAG} \
                --seed "${SEED}" \
                --kv_overrides \
                    task_config.randomized_init=True \
                    task_config.normalized_rl_action_space=False \
                    task_config.obs_goal_horizon=1
        done
        wait
        tb-reducer ./Results/sim/${EXP_NAME}/${TRAIN}/${TAG}/seed* -o ./Results/sim/${EXP_NAME}/${TRAIN}/${TAG}/ -r mean

        # Evaluate the trained controller/agent.
        # RL Experiment
        # wait
        # for EVAL in "${EVAL_LIST[@]}"; do
        #     echo "Evaluating ${EVAL} for ${SYS}_${ALGO} with seed ${SEED}"
        #     if [ "${EVAL}" == 'robustness_ob' ]; then
        #         EXTERNAL_PARAM=(0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
        #     elif [ "${EVAL}" == 'robustness_ps' ]; then
        #         EXTERNAL_PARAM=(0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
        #     elif [ "${EVAL}" == 'robustness_pm' ]; then
        #         EXTERNAL_PARAM=(0 0.05 0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0)
        #     elif [ "${EVAL}" == 'robustness_dw' ]; then
        #         EXTERNAL_PARAM=(1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.5 4.0 4.5 5.0)
        #     elif [ "${EVAL}" == 'generalization' ]; then
        #         EXTERNAL_PARAM=(9 10 11 12 13 14 15)
        #     else
        #         EXTERNAL_PARAM=(1)
        #     fi

        #     for EP in "${EXTERNAL_PARAM[@]}"; do
        #         for SEED in {0..4}; do
        #             python3 ./rlmpc_experiment.py \
        #                 --task ${SYS_NAME} \
        #                 --algo ${ALGO} \
        #                 --overrides \
        #                     ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        #                     "${CONFIG1}" \
        #                 --experiment_type ${EVAL} \
        #                 --seed ${SEED} \
        #                 --kv_overrides \
        #                     algo_config.training=False \
        #                     algo_config.actor_config.mpc_config.jit=False \
        #                     task_config.normalized_rl_action_space=False \
        #                     task_config.randomized_init=True \
        #                     task_config.external_param=${EP} \
        #                     task_config.obs_goal_horizon=10 \
        #                     algo_config.actor_config.mpc_config.n_parallel_solver=1 \
        #                     algo_config.actor_config.mpc_config.n_train_solver=1 \
        #                 --pretrain_path ./Results/sim/${EXP_NAME}/${TRAIN}/${SYS}_${ALGO}_data/seed${SEED}_*/ &
        #         done
        #         wait
        #     done
        #     wait
        # done
        
    done
done
