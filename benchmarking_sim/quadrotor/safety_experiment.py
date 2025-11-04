
import os
import pickle
import sys
from collections import defaultdict
from functools import partial

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, timing

script_path = os.path.dirname(os.path.realpath(__file__))


@timing
def run(gui=False, n_episodes=1, n_steps=None, save_data=True, seed=1):
    '''The main function running experiments for model-based methods.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''
    # Read the additional arguments
    if len(sys.argv) > 1:
        print('sys.argv', sys.argv)
        ALGO = sys.argv[1]
        assert ALGO in ['mpc_acados', 'gpmpc_acados_TP', 'pid', 'ilqr',], \
            f'ALGO {ALGO} not supported. Please choose from [mpc_acados, gpmpc_acados_TP, pid, ilqr].'
    else:
        ALGO = 'mpc_acados'
    CTRL_ADD = ''
    gp_tag = 'safety'
    ADDITIONAL = '_safety'
    SYS = 'quadrotor_2D_attitude'
    TASK = 'tracking'
    PRIOR = '100'
    agent = 'quadrotor' if SYS in ['quadrotor_2D', 'quadrotor_2D_attitude', 'quadrotor_3D_attitude'] else SYS

    # Check if the config file exists
    assert os.path.exists(f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml'), f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml does not exist'
    assert os.path.exists(f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}{CTRL_ADD}.yaml'), f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}{CTRL_ADD}.yaml does not exist'
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', agent,
                    '--overrides',
                    f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml',
                    f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}{CTRL_ADD}.yaml',
                    '--seed', repr(seed),
                    '--use_gpu', 'True',
                    '--output_dir', f'./{ALGO}/results',
                    ]
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    # Merge config and create output directory
    config = fac.merge()
    if ALGO in ['gpmpc_acados', 'gp_mpc', 'gpmpc_acados_TP', 'gpmpc_acados_TRP']:
        num_data_max = config.algo_config.num_epochs * config.algo_config.num_samples
        gp_tag = f'{PRIOR}_{num_data_max}' if gp_tag is None else gp_tag
        config.output_dir = os.path.join(config.output_dir, gp_tag + ADDITIONAL)

    set_dir_from_config(config)
    config.algo_config.output_dir = config.output_dir
    mkdirs(config.output_dir)

    # Create an environment
    env_func = partial(make,
                       config.task,
                       seed=config.seed,
                       **config.task_config
                       )
    random_env = env_func(gui=False)

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                seed=config.seed,
                **config.algo_config
                )

    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes

    # Run the experiment
    for _ in range(n_episodes):
        # Get initial state and create environments
        init_state, _ = random_env.reset()
        static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

        # Create experiment, train, and run evaluation
        if ALGO in ['gpmpc_acados', 'gp_mpc', 'gpmpc_acados_TP', 'gpmpc_acados_TRP']:
            experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
            if config.algo_config.num_epochs == 1:
                print('Evaluating prior controller')
            elif config.algo_config.gp_model_path is not None:
                ctrl.load(config.algo_config.gp_model_path)
            else:
                # Manually launch training
                # (NOTE: not using launch_training method since calling plotting before eval will break the eval)
                experiment.reset()
                train_runs, test_runs = ctrl.learn(env=static_train_env)
        else:
            experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
            experiment.launch_training()

        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

        # Close environments
        static_env.close()
        static_train_env.close()

        # Merge in new trajectory data
        for key, value in trajs_data.items():
            all_trajs[key] += value

    ctrl.close()
    random_env.close()
    metrics = experiment.compute_metrics(all_trajs)
    all_trajs = dict(all_trajs)

    if save_data:
        results = {'trajs_data': all_trajs, 'metrics': metrics}
        with open(f'./{config.output_dir}/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)
        print(f'Results saved to {config.output_dir}')
        # Save rmse to a file
        with open(f'./{config.output_dir}/metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))
    print(f'pyb_client: {ctrl.env.PYB_CLIENT}')


if __name__ == '__main__':
    run(n_episodes=5, seed=2)
