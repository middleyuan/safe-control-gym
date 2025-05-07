import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import munch
from multiprocessing import Pool
from benchmarking_sim.quadrotor.benchmark_util.utils import run_rollouts

notebook_dir = os.path.dirname(os.path.abspath('__file__'))
print('notebook_dir', notebook_dir)

additional = sys.argv[1]
algo = sys.argv[2]
gp_model_tag = sys.argv[3] if len(sys.argv) > 3 else ''
# parallel = True
parallel = False

num_seed = 10
start_seed = 1
seeds = range(start_seed, start_seed + num_seed)


if parallel:
    results = []
    with Pool(processes=3) as pool:
        async_results = [
            pool.apply_async(run_rollouts, args=(munch.munchify({
                'additional': additional,
                'algo': algo,
                'eval_task': 'rollout',
                'start_seed': seed,
                'num_seed': 1,
                'SYS': 'quadrotor_2D_attitude',
                'gp_model_tag': gp_model_tag,
                }),)
            )
            for seed in seeds
        ]
        for async_result in async_results:
            results.append(async_result.get())
else:
    for seed in seeds:
        task_description = munch.munchify({
            'additional': additional,
            'algo': algo,
            'eval_task': 'rollout',
            'start_seed': seed,
            'num_seed': 1,
            # 'SYS': 'quadrotor_3D_attitude',
            'SYS': 'quadrotor_2D_attitude',
            'gp_model_tag': gp_model_tag,
            })
        run_rollouts(task_description)
