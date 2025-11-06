#!/bin/bash

SYS='quadrotor_2D_attitude'
TASK='track'
ALGO='ppo'

SAFETY_FILTER='nl_mpsc'
MPSC_COST='one_step_cost'
FILTER=True
SF_PEN=0.3

# TAG="no_noise"
TAG="noise"

# Train the unsafe controller/agent.
python3 train_rl.py \
    --task quadrotor \
    --algo ${ALGO} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}_${TASK}.yaml \
        ./config_overrides/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SAFETY_FILTER}_${SYS}.yaml \
    --output_dir ./models/rl_models/${TAG}/ \
    --seed 2 \
    --kv_overrides \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.soften_constraints=True \
        algo_config.filter_train_actions=${FILTER} \
        algo_config.penalize_sf_diff=${FILTER} \
        algo_config.sf_penalty=${SF_PEN} \
        task_config.seed=42 \
        algo_config.seed=42 \
        sf_config.seed=42
