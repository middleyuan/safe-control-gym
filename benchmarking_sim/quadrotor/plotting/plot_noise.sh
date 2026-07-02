#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Plot model-based controller OOD robustness test
for noise in 'obs_noise' 'proc_noise' 'param'
# for noise in 'obs_noise' 
# for noise in 'proc_noise'
# for noise in 'param'
do
    echo "Processing model-based controllers for noise type: $noise"
    for algo in 'mpc_acados' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid' 'gpmpc_acados_TP' 'fmpc'
    # for algo in 'fmpc' 
    # for algo in 'pid' 'lqr'
    do
        echo "  Processing $algo..."
        python3 plot_noise.py $algo $noise
    done
    echo ""
done

# Plot RL controller OOD robustness test
for noise in 'obs_noise' 'proc_noise' 'param'
# for noise in 'obs_noise' 
# for noise in 'proc_noise'
# for noise in 'param'
do
    echo "Processing RL controllers for noise type: $noise"
    for algo in 'ppo' 'sac' 'dppo' 'ppo_mpc'
    do
        echo "  Processing $algo..."
        python3 plot_noise_rl.py $algo $noise
    done
    echo ""
done

# Generate combined plots for all methods
for noise in 'obs_noise' 'proc_noise' 'param'
# for noise in 'obs_noise' 
# for noise in 'proc_noise'
# for noise in 'param'
do
    echo "Generating combined plots for noise type: $noise"
    python3 plot_noise_all.py $noise
    echo ""
done
