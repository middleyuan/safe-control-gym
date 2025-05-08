
GP_BASE_TAG='hpo'
START_TIME=$(date +%s)

# noise test
for RAND_TYPE in ''
do
    GP_TAG=$GP_BASE_TAG$RAND_TYPE
    python3 parallel_gpmpc_experiment.py 'gpmpc_acados_TP' $GP_TAG $RAND_TYPE
    python3 results_noise.py 'gpmpc_acados_TP' 'obs_noise' $GP_TAG
    # python3 results_noise.py 'gpmpc_acados_TP' 'proc_noise' $GP_TAG
    # python3 results_param.py 'gpmpc_acados_TP' 'param' $GP_TAG
done

for algo in 'mpc_acados' 'ilqr'
do
    python3 results_noise.py $algo 'obs_noise'
    # python3 results_noise.py $algo 'proc_noise'
    # python3 results_param.py $algo 'param'
done

# # compile results
cd ./plotting
mkdir -p ./noise
for algo in 'mpc_acados' 'ilqr' 'gpmpc_acados_TP'
do
    python3 plot_noise.py $algo 'obs_noise' 
done

# plot all algo
python3 plot_noise_all.py 'obs_noise' 

# python3 ../del_acados_files.py
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Elapsed time: $ELAPSED_TIME seconds"