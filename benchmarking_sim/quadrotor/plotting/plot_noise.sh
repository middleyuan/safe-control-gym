

# plot model based controller OOD robustness test
for noise in 'obs_noise' 'proc_noise' 'param'
do
    for algo in 'mpc_acados' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid' 'gpmpc_acados_TP' 'fmpc'
    do
        python3 plot_noise.py $algo $noise
    done
done

# plot model based controller ID robustness test