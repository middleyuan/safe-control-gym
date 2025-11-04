import sys
import time
from multiprocessing import Pool

import munch
import numpy as np

from benchmarking_sim.quadrotor.benchmark_util.utils import run_rollouts

parallel = False

algo = sys.argv[1]
gp_model_tag = sys.argv[2] if len(sys.argv) > 2 else ''

# Test
additional = '_downwash'
num_seed = 10
start_seed = 1
seeds = range(start_seed, start_seed + num_seed)
dw_height_list = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
                  3.0, 3.5, 4.0, 4.5, 5.0]

time1 = time.perf_counter()
for dw_height in np.arange(1.5, 4.0, 0.1):
    if parallel:
        results = []
        with Pool(processes=3) as pool:
            async_results = [
                pool.apply_async(run_rollouts, args=(munch.munchify({
                    'additional': additional,
                    'algo': algo,
                    'dw_height': dw_height,
                    'eval_task': 'downwash',
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
                'dw_height': dw_height,
                'eval_task': 'downwash',
                'num_seed': 1,
                'start_seed': start_seed,
                'SYS': 'quadrotor_2D_attitude',
                'gp_model_tag': gp_model_tag,
            })
            run_rollouts(task_description)
time2 = time.perf_counter()
print(f'Elapsed time: {time2 - time1:.3f} sec')
