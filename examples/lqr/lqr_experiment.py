'''A LQR and iLQR example.'''

import os
import pickle
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(gui=False, plot=True, n_episodes=1, n_steps=None, save_data=False):
    '''The main function running LQR and iLQR experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        plot (bool): Whether to plot.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    env = env_func(gui=False)

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )

    X_GOAL = np.load('ilqr_ref_traj.npy', allow_pickle=True).item()['obs'][0]
    X_GOAL[:, -2] = 0
    X_GOAL[:, -1] = 0
    env.X_GOAL = X_GOAL
    ctrl.env.X_GOAL = X_GOAL

    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes

    # Run the experiment.
    for _ in range(n_episodes):
        # Create experiment, train, and run evaluation
        experiment = BaseExperiment(env=env, ctrl=ctrl, train_env=env)
        experiment.launch_training()

        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

        if plot is True:
            post_analysis(trajs_data['obs'][0], trajs_data['action'][0], ctrl.env)

        # Merge in new trajectory data
        for key, value in trajs_data.items():
            all_trajs[key] += value

    ctrl.close()
    env.close()
    metrics = experiment.compute_metrics(all_trajs)
    all_trajs = dict(all_trajs)
    # np.save('./ilqr_ref_traj.npy', all_trajs, allow_pickle=True)

    if save_data:
        results = {'trajs_data': all_trajs, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))


def post_analysis(state_stack, input_stack, env):
    '''Plots the input and states to determine iLQR's success.

    Args:
        state_stack (ndarray): The list of observations of iLQR in the latest run.
        input_stack (ndarray): The list of inputs of iLQR in the latest run.
    '''

    _, ax = plt.subplots()
    ax.plot(state_stack[:, 0], state_stack[:, 2], label='iLQR')
    ax.plot(env.X_GOAL[:, 0], env.X_GOAL[:, 2], label='Ref')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    run()
