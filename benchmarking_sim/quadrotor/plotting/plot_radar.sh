
# for algo in 'iLQR' 'PID' 'LQR' \
#             'GP-MPC' 'Nonlinear-MPC' 'Linear-MPC' 'F-MPC' \
#             'PPO' 'SAC' 'DPPO' 'PPO-MPC' # 'PPO-ID' 'SAC-ID' 'DPPO-ID'
# for algo in  'PPO-ID' 'SAC-ID' 'DPPO-ID'
for algo in 'ilqr' 'pid' 'lqr' \
            'gpmpc_acados_TP' 'mpc_acados' 'linear_mpc_acados' 'fmpc' \
            'ppo' 'sac' 'dppo' 'ppo_mpc' # 'ppo_id' 'sac_id' 'dppo_id'
do
    python3 plot_radar.py $algo
    python3 plot_radar.py $algo 'abs'
done
