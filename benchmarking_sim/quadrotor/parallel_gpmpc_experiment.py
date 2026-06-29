import os
import sys
import numpy as np
from multiprocessing import Pool
from benchmarking_sim.quadrotor.mb_experiment import run

script_path = os.path.dirname(os.path.realpath(__file__))

def run_experiment(seed, algo, gp_tag, additional, control_add='', output_root='Results'):
    try:
        sys.argv[1:] = [algo, gp_tag, additional, control_add, output_root]
        run(seed=seed)
        return (seed, run.elapsed_time, None)  # Return seed, runtime, and no error
    except Exception as e:
        # Capture error details
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error_message = f'Error: {e}\n{exc_type} {fname} {exc_tb.tb_lineno}\n'
        # Dump the error to a file
        with open(f'./error_{seed}.txt', 'w') as f:
            f.write(error_message)
        return (seed, None, error_message)  # Return seed, no runtime, and error message

if __name__ == '__main__':
    ALGO = sys.argv[1]
    GP_TAG = sys.argv[2] if len(sys.argv) > 2 else 'handtuned'
    ADDITIONAL = sys.argv[3] if len(sys.argv) > 3 else ''
    CTRL_ADD = sys.argv[4] if len(sys.argv) > 4 else ''
    OUTPUT_ROOT = sys.argv[5] if len(sys.argv) > 5 else 'Results'

    num_seed = 10
    start_seed = 1
    seeds = range(start_seed, start_seed + num_seed)

    results = []
    
    parallel = True
    # parallel = False  # Set to True to run in parallel
    if parallel:
        # Run experiments in parallel
        with Pool(processes=5) as pool:
            async_results = [
                pool.apply_async(run_experiment, args=(seed, ALGO, GP_TAG, ADDITIONAL, CTRL_ADD, OUTPUT_ROOT))
                for seed in seeds
            ]

            for async_result in async_results:
                results.append(async_result.get())
    else:
        # Run experiments sequentially
        for seed in seeds:
            results.append(run_experiment(seed, ALGO, GP_TAG, ADDITIONAL, CTRL_ADD, OUTPUT_ROOT))

    # Process results
    runtime_list = []
    suceeded = 0
    for seed, runtime, error in results:
        if error is None:
            runtime_list.append(runtime)
            suceeded += 1
        else:
            print(f'Seed {seed} failed with error:\n{error}')

    print(f'{suceeded} out of {num_seed} runs succeeded')
    if runtime_list:
        print(f'Average runtime for {suceeded} runs: {np.mean(runtime_list):.3f} sec')
