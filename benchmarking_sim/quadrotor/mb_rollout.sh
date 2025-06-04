
# GP_BASE_TAG='100_300'
# GP_BASE_TAG='100_300'
GP_BASE_TAG='hpo'

# get the time
START_TIME=$(date +%s)

# # param
# for RAND_TYPE in '_param'
# do
#     GP_TAG=$GP_BASE_TAG$RAND_TYPE
#     # python3 parallel_gpmpc_experiment.py 'gpmpc_acados_TP' $GP_BASE_TAG $RAND_TYPE $RAND_TYPE
#     python3 results_param.py 'gpmpc_acados_TP' 'param' $GP_TAG
# done

# # task
# for RAND_TYPE in '_tr' 
# do
#     GP_TAG=$GP_BASE_TAG$RAND_TYPE
#     python3 parallel_gpmpc_experiment.py 'gpmpc_acados_TP'  $GP_BASE_TAG $RAND_TYPE $RAND_TYPE
#     for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
#     do
#         for algo in 'gpmpc_acados_TP'
#         do
#             python3 results_rollout.py $ADDITIOANL $STARTSEED $algo $GP_TAG
#         done
#     done
# done

# # noise
# for RAND_TYPE in '_ob_ns=5_proc_ns=3' \
#                  '_ob_ns=5'\
#                  '_proc_ns=3'
# do
#     GP_TAG=$GP_BASE_TAG$RAND_TYPE
#     python3 parallel_gpmpc_experiment.py 'gpmpc_acados_TP' $RAND_TYPE
#     python3 results_noise.py 'gpmpc_acados_TP' 'obs_noise' $GP_TAG
#     python3 results_noise.py 'gpmpc_acados_TP' 'proc_noise' $GP_TAG
# done

# # nominal
# for RAND_TYPE in ''
# do
#     GP_TAG=$GP_BASE_TAG$RAND_TYPE
#     python3 parallel_gpmpc_experiment.py 'gpmpc_acados_TP' $GP_BASE_TAG $RAND_TYPE
#     # python3 results_dw.py 'gpmpc_acados_TP' $GP_TAG
#     # python3 results_noise.py 'gpmpc_acados_TP' 'obs_noise' $GP_TAG
#     # python3 results_noise.py 'gpmpc_acados_TP' 'proc_noise' $GP_TAG
#     # python3 results_param.py 'gpmpc_acados_TP' 'param' $GP_TAG
#     for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
#     do
#         for algo in 'gpmpc_acados_TP'
#         do
#             python3 results_rollout.py $ADDITIOANL $STARTSEED $algo $GP_TAG
#         done
#     done
# done

# for RAND_TYPE in '_ob_ns=5_proc_ns=3' \
#                  '_ob_ns=5'
# do
#     GP_TAG=$GP_BASE_TAG$RAND_TYPE
#     python3 results_noise.py 'gpmpc_acados_TP' 'obs_noise' $GP_TAG
# done

# for RAND_TYPE in '_ob_ns=5_proc_ns=3' \
#                  '_proc_ns=3'
# do
#     GP_TAG=$GP_BASE_TAG$RAND_TYPE
#     python3 results_noise.py 'gpmpc_acados_TP' 'proc_noise' $GP_TAG
# done

# model-based
# for algo in 'mpc_acados' 'fmpc' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid'
# for algo in 'fmpc'
# do
#     for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
#     do
#         python3 results_rollout.py $ADDITIOANL $algo 
#     done
#     python3 extract_results.py $algo
# done

# for algo in 'lqr' 'ilqr' 'pid' 'mpc_acados' 'linear_mpc_acados' 'fmpc' 
for algo in 'fmpc'
do
    python3 results_noise.py $algo 'obs_noise'
    python3 results_noise.py $algo 'proc_noise'
    python3 results_param.py $algo 'param'
    python3 results_dw.py $algo
done

# python3 ../del_acados_files.py
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Elapsed time: $ELAPSED_TIME seconds"