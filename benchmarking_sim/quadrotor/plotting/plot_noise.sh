

# plot model based controller OOD robustness test
# for noise in 'obs_noise' 'proc_noise' 'param'
for noise in 'obs_noise' 
# for noise in 'proc_noise'
# for noise in 'param'
do
    for algo in 'mpc_acados' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid' 'gpmpc_acados_TP' 'fmpc'
    # for algo in 'linear_mpc_acados' 
    do
        python3 plot_noise.py $algo $noise
    done
done

for noise in 'obs_noise' 'proc_noise' 'param'
# for noise in 'obs_noise' 
# for noise in 'proc_noise'
# for noise in 'param'
do
    python3 plot_noise_all.py $noise
done

