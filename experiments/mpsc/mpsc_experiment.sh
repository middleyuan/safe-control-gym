#!/bin/bash

SYS='quadrotor_2D_attitude'
TASK='tracking'
ALGO='ppo'

SAFETY_FILTER='nl_mpsc'
MPSC_COST='precomputed_cost'
MPSC_COST_HORIZON=25
DECAY_FACTOR=1

python3 ./mpsc_experiment.py \
    --task quadrotor \
    --algo ${ALGO} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}_${TASK}.yaml \
        ./config_overrides/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SAFETY_FILTER}_${SYS}.yaml \
    --kv_overrides \
        sf_config.cost_function=${MPSC_COST} \
        sf_config.mpsc_cost_horizon=${MPSC_COST_HORIZON} \
        sf_config.decay_factor=${DECAY_FACTOR} \
        sf_config.max_w=0.0 \
        sf_config.slack_cost=1000.0 \
        task_config.seed=42 \
        algo_config.seed=42 \
        sf_config.seed=42
