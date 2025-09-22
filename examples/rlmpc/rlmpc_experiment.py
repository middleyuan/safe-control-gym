"""This script tests the RL implementation."""

import shutil
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import pickle

from safe_control_gym.envs.benchmark_env import Environment, Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(gui=False, plot=True, n_episodes=10, n_steps=None, curr_path='.'):
    """Main function to run RL experiments.

    Args:
        gui (bool): Whether to display the gui.
        plot (bool): Whether to plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.

    Returns:
        X_GOAL (np.ndarray): The goal (stabilization or reference trajectory) of the experiment.
        results (dict): The results of the experiment.
        metrics (dict): The metrics of the experiment.
    """

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    config.seed += 150

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    # Experiment settings
    if config.experiment_type == 'robustness_ob':
        config.task_config.disturbances.observation[0].std = [
            config.task_config.external_param*i for i in config.task_config.disturbances.observation[0].std
        ]
    elif config.experiment_type == 'robustness_ps':
        config.task_config.disturbances.action[0].std = [
            config.task_config.external_param*i for i in config.task_config.disturbances.action[0].std
        ]
    elif config.experiment_type == 'robustness_pm':
        config.task_config.randomized_inertial_prop = True
        for p in config.task_config.inertial_prop_randomization_info.keys():
            config.task_config.inertial_prop_randomization_info[p]['scale'] *= config.task_config.external_param
    elif config.experiment_type == 'robustness_dw':
        config.task_config.disturbances.downwash[0].pos[2] = config.task_config.external_param
    elif config.experiment_type == 'generalization':
        config.task_config.episode_len_sec = config.task_config.external_param
        config.task_config.task_info.pop('ilqr_traj_data', None)

    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func(gui=gui)

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir=curr_path + '/temp')

    # Load state_dict from trained.
    # ctrl.load(f'{curr_path}/models/{config.algo}/{config.algo}_model_{system}_{task}.pt')
    # ctrl.load(f'{curr_path}/models/{config.algo}/model_latest.pt')
    if 'pretrain_path' in config.keys():
        # ctrl.load(config.pretrain_path + "model_latest.pt")
        ctrl.load(config.pretrain_path + "model_best.pt")
    else:
        pass
        # ctrl.load(f'{curr_path}/models/{config.algo}/model_best_7.pt')
        # dummy_param = np.array([-238.1, -21.35, 179.65, -238.1, -21.35, 179.65, -170.4, -22.22, 280, 0.0905, 0.8, 0.0814])
        # print(ctrl.agent.ac.actor.mpc_param)
        # print(ctrl.agent.ac.actor.mpc_param[-12:].detach().numpy()*dummy_param)

    # Remove temporary files and directories
    shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)

    # Run experiment
    experiment = BaseExperiment(env, ctrl)
    results, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    ctrl.close()

    ### Housekeeping
    if config.experiment_type == "performance":
        temp = config.pretrain_path+"/perf_metric.npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "generalization":
        metrics['episode_len_sec'] = config.task_config.external_param
        temp = config.pretrain_path+"/transfer_metric_"+str(config.task_config.external_param)+".npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "robustness_ob":
        metrics['noise_scale'] = config.task_config.external_param
        temp = config.pretrain_path+"/robust_metric_ob_"+str(config.task_config.external_param)+".npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "robustness_ps":
        metrics['noise_scale'] = config.task_config.external_param
        temp = config.pretrain_path+"/robust_metric_ps_"+str(config.task_config.external_param)+".npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "robustness_pm":
        metrics['noise_scale'] = config.task_config.external_param
        temp = config.pretrain_path+"/robust_metric_pm_"+str(config.task_config.external_param)+".npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "robustness_dw":
        metrics['downwash_height'] = config.task_config.external_param
        temp = config.pretrain_path+"/robust_metric_dw_"+str(config.task_config.external_param)+".npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "traj_data":
        temp = f"./traj_results_{config.algo}_{config.task_config.episode_len_sec}.npy"
        if config.seed-150 == 0:  # os.path.isfile(temp):
            data = {'n_rollouts': n_episodes,
                    'obs': np.array(results['obs']),
                    'timestamp': np.array(results['timestamp'])}
        else:
            data = np.load(temp, allow_pickle=True).item()
            data['n_rollouts'] += n_episodes
            data['obs'] = np.concatenate((data['obs'], np.array(results['obs'])), axis=0)
            data['timestamp'] = np.concatenate((data['timestamp'], np.array(results['timestamp'])), axis=0)
        np.save(temp, data, allow_pickle=True)
    print(metrics)
    # with open(f'./ppo_mpc_safety_config_results.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    if plot is True:
        if system == Environment.CARTPOLE:
            graph1_1 = 2
            graph1_2 = 3
            graph3_1 = 0
            graph3_2 = 1
        elif system == 'quadrotor_2D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
        elif system == 'quadrotor_3D':
            graph1_1 = 6
            graph1_2 = 9
            graph3_1 = 0
            graph3_2 = 4
        elif system == 'quadrotor_4D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
        elif system == 'quadrotor_9D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2

        _, ax3 = plt.subplots()
        ax3.plot(results['obs'][0][:, graph3_1], results['obs'][0][:, graph3_2], 'r--', label='RL Trajectory')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(env.X_GOAL[:, graph3_1], env.X_GOAL[:, graph3_2], 'g--', label='Reference')
        ax3.scatter(results['obs'][0][0, graph3_1], results['obs'][0][0, graph3_2], color='g', marker='o', s=100,
                    label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        post_analysis(results['obs'][0], results['action'][0], env)
        # plt.savefig(f"{curr_path}/perf.png")

    return env.X_GOAL, results, metrics


def post_analysis(state_stack, input_stack, env):
    '''Plots the input and states to determine iLQR's success.

    Args:
        state_stack (ndarray): The list of observations of iLQR in the latest run.
        input_stack (ndarray): The list of inputs of iLQR in the latest run.
    '''
    model = env.symbolic
    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Plot states
    fig, axs = plt.subplots(model.nx)
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    # Plot inputs
    _, axs = plt.subplots(model.nu)
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        axs[k].set(ylabel=f'input {k}')
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')

    plt.show()


if __name__ == '__main__':
    run()
