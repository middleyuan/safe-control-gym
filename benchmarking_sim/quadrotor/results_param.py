
import sys
import time
import munch
import numpy as np
from multiprocessing import Pool
from benchmarking_sim.quadrotor.benchmark_util.utils import run_rollouts

parallel = False
# parallel = True

algo = sys.argv[1]
# algo = 'gpmpc_acados_TP'
# algo = 'linear_mpc_acados'
# algo = 'mpc_acados'
# algo = 'lqr'
noise_type = sys.argv[2] if len(sys.argv) > 2 else 'param'
gp_model_tag = sys.argv[3] if len(sys.argv) > 3 else ''

# noise factor test
additional = '_11'
# noise_factor_list = [0,1,2,3,4,5,10,15,20,25] #,\
                    #  30,40,50,60,70,80,90,100]
noise_factor_list = np.arange(0, 5.0, 0.2)
num_seed = 10
start_seed = 1
seeds = range(start_seed, start_seed + num_seed)

time1 = time.perf_counter()
for noise_factor in noise_factor_list:
    if parallel:
        results = []
        with Pool(processes=5) as pool:
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
                })
            run_rollouts(task_description)
time2 = time.perf_counter()
print(f'Elapsed time: {time2 - time1:.3f} sec')
    