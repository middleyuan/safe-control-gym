"""This script tests the RL implementation."""

import shutil
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Environment, Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(gui=False, plot=True, n_episodes=10, n_steps=None, curr_path='.', model_src_dir=None):
    """Main function to run RL experiments.

    Args:
        gui (bool): Whether to display the gui.
        plot (bool): Whether to plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.
        model_src_dir (str): Directory containing model_best.pt and config file to copy.

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
        ctrl.load(f'{curr_path}/models/{config.algo}/model_best.pt')

    # Remove temporary files and directories
    shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)

    # Run experiment
    experiment = BaseExperiment(env, ctrl)
    results, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    ctrl.close()

    ### Housekeeping
    data_storage_path = '.'
    if 'pretrain_path' in config.keys():
        data_storage_path = config.pretrain_path
    if config.experiment_type == "performance":
        temp = data_storage_path+"/perf_metric.npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "generalization":
        metrics['episode_len_sec'] = config.task_config.external_param
        temp = data_storage_path+"/transfer_metric_"+str(config.task_config.external_param)+".npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "robustness_ob":
        metrics['noise_scale'] = config.task_config.external_param
        temp = data_storage_path+"/robust_metric_ob_"+str(config.task_config.external_param)+".npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "robustness_ps":
        metrics['noise_scale'] = config.task_config.external_param
        temp = data_storage_path+"/robust_metric_ps_"+str(config.task_config.external_param)+".npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "robustness_pm":
        metrics['noise_scale'] = config.task_config.external_param
        temp = data_storage_path+"/robust_metric_pm_"+str(config.task_config.external_param)+".npy"
        np.save(temp, metrics, allow_pickle=True)
    elif config.experiment_type == "robustness_dw":
        metrics['downwash_height'] = config.task_config.external_param
        temp = data_storage_path+"/robust_metric_dw_"+str(config.task_config.external_param)+".npy"
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

    if plot is False:
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
        elif system == 'quadrotor_6D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
            graph3_3 = 4
        elif system == 'quadrotor_9D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
            graph3_3 = 4

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
            ax3.set_ylabel(r'Y')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')
        plt.savefig(f"{curr_path}/trajectory_xy.png")  # Save the figure
        actual_traj = results['obs'][0][:, [graph3_1, graph3_2]]
        ref_traj = env.X_GOAL[:, [graph3_1, graph3_2]]
        # Ensure they have the same number of time steps
        print(len(actual_traj), len(ref_traj))
        ref_traj = ref_traj[:len(actual_traj)] 
        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual_traj - ref_traj) ** 2))
        print(f"Trajectory RMSE: {rmse:.4f}")
        print((actual_traj - ref_traj))
        
        diff = actual_traj - ref_traj
        time_steps = range(len(diff))

        # actual_traj = results['obs'][0][:, [graph3_1, graph3_2]]
        # ref_traj = env.X_GOAL[:, [graph3_1, graph3_2]]
        # # Ensure they have the same number of time steps
        # print(len(actual_traj), len(ref_traj))
        # # min_len = min(len(actual_traj), len(ref_traj))
        # # actual_traj = actual_traj[:min_len]
        # # ref_traj = ref_traj[:min_len]
        # ref_traj = ref_traj[1:]
        # # Calculate RMSE
        # rmse = np.sqrt(np.mean((actual_traj - ref_traj) ** 2))
        # print(f"Trajectory RMSE: {rmse:.4f}")
        # print((actual_traj - ref_traj))
        #
        # diff = actual_traj - ref_traj
        # time_steps = range(len(diff))
        #
        # plt.figure(figsize=(10, 5))
        # plt.plot(time_steps, diff[:, 0], label='X difference')
        # plt.plot(time_steps, diff[:, 1], label='Y difference')
        # plt.xlabel('Time step')
        # plt.ylabel('Difference')
        # plt.title('Trajectory Differences Over Time')
        # plt.legend()
        # plt.grid(True)
        # errors = np.linalg.norm(actual_traj - ref_traj, axis=1)  # Euclidean distance at each step
        # # plt.show()
        # plt.savefig(f"{curr_path}/trajectory_diff.png")  # Save instead of show        errors = np.linalg.norm(actual_traj - ref_traj, axis=1)  # Euclidean distance at each step
        # rmse = np.sqrt(np.mean(errors**2))
        # print(f"2ndTrajectory RMSE: {rmse:.4f}")
        #
        #
        # plt.figure(figsize=(10, 4))
        # plt.plot(range(len(results['obs'][0])), results['obs'][0][:, graph3_3], label='Z trajectory', color='blue')
        # if config.task == Environment.QUADROTOR:
        #     plt.plot(range(len(env.X_GOAL)), env.X_GOAL[:, graph3_3], label='Z reference', color='green', linestyle='--')
        # plt.xlabel('Time step')
        # plt.ylabel('Z position')
        # plt.title('Z Position Over Time')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        # plt.savefig(f"{curr_path}/z_position.png")  # Save instead of show
        

        post_analysis(results['obs'][0], results['action'][0], env, curr_path)

    return env.X_GOAL, results, metrics


def post_analysis(state_stack, input_stack, env, curr_path):
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
    plt.savefig(f"{curr_path}/state_stats.png")
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
    # plt.savefig(f"{curr_path}/input_stats.png")


if __name__ == '__main__':
    run()