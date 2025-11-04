
import sys
import time
from multiprocessing import Pool

import munch

from benchmarking_sim.quadrotor.benchmark_util.utils import run_rollouts

parallel = False

algo = sys.argv[1]
noise_type = sys.argv[2] if len(sys.argv) > 2 else 'param'
gp_model_tag = sys.argv[3] if len(sys.argv) > 3 else ''

# Noise factor test
additional = '_11'
noise_factor_list = [0, 0.01, 0.02, 0.05, 0.1,
                     0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
                     1.4, 1.6, 1.8, 2.0, 2.2, 2.4,
                     2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0]
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
        for start_seed in seeds:
            task_description = munch.munchify({
                'additional': additional,
                'algo': algo,
                'noise_factor': noise_factor,
                'eval_task': noise_type,
                'num_seed': 1,
                'start_seed': start_seed,
                'SYS': 'quadrotor_2D_attitude',
                'gp_model_tag': gp_model_tag,
            })
            run_rollouts(task_description)
time2 = time.perf_counter()
print(f'Elapsed time: {time2 - time1:.3f} sec')
