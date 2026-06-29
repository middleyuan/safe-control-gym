
import sys
import time
import munch
import numpy as np
from multiprocessing import Pool
from benchmarking_sim.quadrotor.benchmark_util.utils import run_rollouts

parallel = True

algo = sys.argv[1]
# algo = 'gpmpc_acados_TP'
# algo = 'linear_mpc_acados'
# algo = 'mpc_acados'
# algo = 'lqr'
noise_type = sys.argv[2] if len(sys.argv) > 2 else 'param'
gp_model_tag = sys.argv[3] if len(sys.argv) > 3 else ''
output_root = sys.argv[4] if len(sys.argv) > 4 else 'Results'

# noise factor test
additional = '_11'
# noise_factor_list = [0,1,2,3,4,5,10,15,20,25] #,\
                    #  30,40,50,60,70,80,90,100]
# noise_factor_list = np.arange(0, 5.0, 0.2)
# noise_factor_list = [0, 0.01, 0.02, 0.05, 0.1, \
#                      0.2, 0.4, 0.6, 0.8, 1.0, 1.2, \
#                      1.4, 1.6, 1.8, 2.0, 2.2, 2.4, \
#                      2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0]
noise_factor_list = [0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0,
                     2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0,
                     7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                     14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                     20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
num_seed = 10
start_seed = 1
seeds = range(start_seed, start_seed + num_seed)
is_gp_controller = algo in ['gpmpc_acados', 'gpmpc_acados_TP', 'gpmpc_acados_TRP', 'gp_mpc']
num_processes = 3 if is_gp_controller else 10

time1 = time.perf_counter()
for noise_factor in noise_factor_list:
    if parallel:
        results = []
        with Pool(processes=num_processes) as pool:
            async_results = [
                pool.apply_async(run_rollouts, args=(munch.munchify({
                    'additional': additional,
                    'algo': algo,
                    'noise_factor': noise_factor,
                    'eval_task': noise_type,
                    'num_seed': 1,
                    'start_seed': seed,
                    'SYS': 'quadrotor_2D_attitude', 
                    'gp_model_tag': gp_model_tag,
                    'output_root': output_root,
                    'exp_name': noise_type,
                    }),)
                )
                for seed in seeds
            ]
            for async_result in async_results:
                results.append(async_result.get())
    else:
    # if True:
        for start_seed in seeds:
            task_description = munch.munchify({
                'additional': additional,
                'algo': algo,
                'noise_factor': noise_factor,
                'eval_task': noise_type,
                'num_seed': 1,
                'start_seed': start_seed,
                # 'SYS': 'quadrotor_3D_attitude',
                'SYS': 'quadrotor_2D_attitude', 
                'gp_model_tag': gp_model_tag,
                'output_root': output_root,
                'exp_name': noise_type,
                })
            run_rollouts(task_description)
time2 = time.perf_counter()
print(f'Elapsed time: {time2 - time1:.3f} sec')
