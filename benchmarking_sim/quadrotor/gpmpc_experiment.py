
import os
import sys
import time

from benchmarking_sim.quadrotor.mb_experiment import run

script_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    ALGO = sys.argv[1]
    ADDITIONAL = sys.argv[2] if len(sys.argv) > 2 else ''

    runtime_list = []
    num_seed = 10
    start_seed = 1
    suceeded = 0
    seed = start_seed

    time1 = time.perf_counter()
    while suceeded < num_seed:
        if seed > num_seed + start_seed:
            print(f'{suceeded} out of {num_seed} runs succeeded')
            break
        try:
            sys.argv[1:] = [ALGO,
                            ADDITIONAL,
                            ]
            run(seed=seed)
            runtime_list.append(run.elapsed_time)
            suceeded += 1
        except Exception as e:
            # Log the error and complete trackback
            print('Error:', e)
            # The error and complete trackback
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            # Dump the error to a file
            with open(f'./error_{seed}.txt', 'w') as f:
                f.write(f'Error: {e}\n')
                f.write(f'{exc_type} {fname} {exc_tb.tb_lineno}\n')
        seed += 1

    time2 = time.perf_counter()
    print(f'Elapsed time: {time2 - time1:.3f} sec')
