
GP_BASE_TAG='hpo'

# Get the time
START_TIME=$(date +%s)

# Nominal
for RAND_TYPE in ''
do
    GP_TAG=$GP_BASE_TAG$RAND_TYPE
    python3 results_dw.py 'gpmpc_acados_TP' $GP_TAG
    python3 results_noise.py 'gpmpc_acados_TP' 'obs_noise' $GP_TAG
    python3 results_noise.py 'gpmpc_acados_TP' 'proc_noise' $GP_TAG
    python3 results_param.py 'gpmpc_acados_TP' 'param' $GP_TAG
    for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
    do
        for algo in 'gpmpc_acados_TP'
        do
            python3 results_rollout.py $ADDITIOANL $STARTSEED $algo $GP_TAG
        done
    done
done

# Model-based
for algo in 'mpc_acados' 'fmpc' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid'
do
    for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
    do
        python3 results_rollout.py $ADDITIOANL $algo
    done
    python3 extract_results.py $algo
done

for algo in 'lqr' 'ilqr' 'pid' 'mpc_acados' 'linear_mpc_acados' 'fmpc'
do
    python3 results_noise.py $algo 'obs_noise'
    python3 results_noise.py $algo 'proc_noise'
    python3 results_param.py $algo 'param'
    python3 results_dw.py $algo
done

# Delete acados files
python3 ../del_acados_files.py
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Elapsed time: $ELAPSED_TIME seconds"
