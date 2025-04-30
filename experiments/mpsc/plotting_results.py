'''This script analyzes and plots the results from MPSC experiments.'''

import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.experiments.base_experiment import MetricExtractor
from safe_control_gym.safety_filters.mpsc.mpsc_utils import get_discrete_derivative
from safe_control_gym.utils.plotting import load_from_logs

plot = True  # Saves figure if False

U_EQ = np.array([0.3, 0])

met = MetricExtractor()
met.verbose = False


def load_all_models(system, task, algo):
    '''Loads the results of every experiment.

    Args:
        system (str): The system to be plotted.
        task (str): The task to be plotted (either 'stab' or 'track').
        algo (str): The controller to be plotted.

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    all_results = {}

    for model in ordered_models:
        with open(f'./results_mpsc/{model}.pkl', 'rb') as f:
            all_results[model] = pickle.load(f)

    return all_results


def extract_magnitude_of_corrections(results_data):
    '''Extracts the magnitude of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        magn_of_corrections (list): The list of magnitude of corrections for all experiments.
    '''

    magn_of_corrections = [np.linalg.norm(mpsc_results['correction'][0]) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    return magn_of_corrections


def extract_max_correction(results_data):
    '''Extracts the max correction from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        max_corrections (list): The list of max corrections for all experiments.
    '''
    max_corrections = [np.max(np.abs(mpsc_results['correction'][0])) for mpsc_results in results_data['cert_results']['safety_filter_data']]

    return max_corrections


def extract_number_of_corrections(results_data):
    '''Extracts the number of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        num_corrections (list): The list of the number of corrections for all experiments.
    '''
    num_corrections = [np.sum(mpsc_results['correction'][0] * 10.0 > np.linalg.norm(results_data['cert_results']['current_clipped_action'][i] - U_EQ, axis=1)) for i, mpsc_results in enumerate(results_data['cert_results']['safety_filter_data'])]
    return num_corrections


def extract_feasible_iterations(results_data):
    '''Extracts the number of feasible iterations from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        feasible_iterations (list): The list of the number of feasible iterations for all experiments.
    '''
    feasible_iterations = [np.sum(mpsc_results['feasible'][0]) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    return feasible_iterations


def extract_rmse(results_data, certified=True):
    '''Extracts the RMSEs from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        rmse (list): The list of RMSEs for all experiments.
    '''
    if certified:
        met.data = results_data['cert_results']
        rmse = np.asarray(met.get_episode_rmse())
    else:
        met.data = results_data['uncert_results']
        rmse = np.asarray(met.get_episode_rmse())
    return rmse


def extract_length(results_data, certified=True):
    '''Extracts the lengths from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        length (list): The list of lengths for all experiments.
    '''
    if certified:
        met.data = results_data['cert_results']
        length = np.asarray(met.get_episode_lengths())
    else:
        met.data = results_data['uncert_results']
        length = np.asarray(met.get_episode_lengths())
    return length


def extract_simulation_time(results_data, certified=True):
    '''Extracts the simulation time from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        sim_time (list): The list of simulation times for all experiments.
    '''
    if certified:
        sim_time = [timestamp[-1] - timestamp[0] for timestamp in results_data['cert_results']['timestamp']]
    else:
        sim_time = [timestamp[-1] - timestamp[0] for timestamp in results_data['uncert_results']['timestamp']]

    return sim_time


def extract_constraint_violations(results_data, certified=True):
    '''Extracts the simulation time from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        num_violations (list): The list of number of constraint violations for all experiments.
    '''
    if certified:
        met.data = results_data['cert_results']
        num_violations = np.asarray(met.get_episode_constraint_violation_steps())
    else:
        met.data = results_data['uncert_results']
        num_violations = np.asarray(met.get_episode_constraint_violation_steps())

    return num_violations


def extract_rate_of_change(results_data, certified=True, order=1, mode='input'):
    '''Extracts the rate of change of a signal from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.
        order (int): Either 1 or 2, denoting the order of the derivative.
        mode (string): Either 'input' or 'correction', denoting which signal to use.

    Returns:
        roc (list): The list of rate of changes.
    '''
    n = min(results_data['cert_results']['current_clipped_action'][0].shape)

    if mode == 'input':
        if certified:
            all_signals = [actions - U_EQ for actions in results_data['cert_results']['current_clipped_action']]
        else:
            all_signals = [actions - U_EQ for actions in results_data['uncert_results']['current_clipped_action']]
    elif mode == 'correction':
        all_signals = [np.squeeze(mpsc_results['uncertified_action'][0]) - np.squeeze(mpsc_results['certified_action'][0]) for mpsc_results in results_data['cert_results']['safety_filter_data']]

    total_derivatives = []
    for signal in all_signals:
        if n == 1:
            ctrl_freq = 15
            if mode == 'correction':
                signal = np.atleast_2d(signal).T
        elif n > 1:
            ctrl_freq = 50
        derivative = get_discrete_derivative(signal, ctrl_freq)
        if order == 2:
            derivative = get_discrete_derivative(derivative, ctrl_freq)
        total_derivatives.append(np.linalg.norm(derivative, 'fro'))

    return total_derivatives


def extract_reward(results_data, certified):
    '''Extracts the mean reward from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        mean_reward (list): The list of mean rewards.
    '''
    if certified:
        met.data = results_data['cert_results']
        returns = np.asarray(met.get_episode_returns())
    else:
        met.data = results_data['uncert_results']
        returns = np.asarray(met.get_episode_returns())

    return returns


def extract_failed(results_data, certified):
    '''Extracts the percent failed from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        failed (list): The percent failed.
    '''
    if certified:
        data = results_data['cert_results']
    else:
        data = results_data['uncert_results']

    failed = [data['info'][i][-1]['out_of_bounds'] for i in range(len(data['info']))]

    return [np.mean(failed)]


def plot_model_comparisons(system, task, algo, data_extractor):
    '''Plots the constraint violations of every controller for a specific experiment.

    Args:
        system (str): The system to be plotted.
        task (str): The task to be plotted (either 'stab' or 'track').
        algo (str): The controller to be plotted.
        data_extractor (func): The function which extracts the desired data.
    '''

    all_results = load_all_models(system, task, algo)

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = ordered_models

    data = []

    for model in ordered_models:
        exp_data = all_results[model]
        data.append(data_extractor(exp_data))

    ylabel = data_extractor.__name__.replace('extract_', '').replace('_', ' ').title()
    ax.set_ylabel(ylabel, weight='bold', fontsize=45, labelpad=10)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x, labels, weight='bold', fontsize=15, rotation=30, ha='right')

    medianprops = dict(linestyle='--', linewidth=2.5, color='black')
    bplot = ax.boxplot(data, patch_artist=True, labels=labels, medianprops=medianprops, widths=[0.75] * len(labels), showfliers=False)

    for patch, color in zip(bplot['boxes'], colors.values()):
        patch.set_facecolor(color)

    fig.tight_layout()
    ax.set_ylim(ymin=0)

    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    else:
        image_suffix = data_extractor.__name__.replace('extract_', '')
        fig.savefig(f'./results_mpsc/{image_suffix}.png', dpi=300)
    plt.close()


def normalize_actions(actions):
    '''Normalizes an array of actions.

    Args:
        actions (ndarray): The actions to be normalized.

    Returns:
        normalized_actions (ndarray): The normalized actions.
    '''
    if system_name == 'cartpole':
        action_scale = 10.0
        normalized_actions = actions / action_scale
    elif system_name == 'quadrotor_2D':
        hover_thrust = 0.1323
        norm_act_scale = 0.1
        normalized_actions = (actions / hover_thrust - 1.0) / norm_act_scale
    else:
        hover_thrust = 0.06615
        norm_act_scale = 0.1
        normalized_actions = (actions / hover_thrust - 1.0) / norm_act_scale

    return normalized_actions


def plot_all_logs(system, task, algo):
    '''Plots comparative plots of all the logs.

    Args:
        system (str): The system to be plotted.
        task (str): The task to be plotted (either 'stab' or 'track').
        algo (str): The controller to be plotted.
    '''
    all_results = {}

    for model in ordered_models:
        all_results[model] = []
        all_results[model].append(load_from_logs(f'./models/rl_models/{model}/logs/'))

    for key in all_results[ordered_models[0]][0].keys():
        if key == 'stat_eval/ep_return':
            plot_log(key, all_results)
        if key == 'stat/constraint_violation':
            plot_log(key, all_results)


def plot_log(key, all_results):
    '''Plots a comparative plot of the log 'key'.

    Args:
        key (str): The name of the log to be plotted.
        all_results (dict): A dictionary of all the logged results for all models.
    '''
    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = ordered_models

    for model, label in zip(ordered_models, labels):
        x = all_results[model][0][key][1] / 1000
        all_data = np.array([values[key][3] for values in all_results[model]])
        ax.plot(x, np.mean(all_data, axis=0), label=label, color=colors[model])
        ax.fill_between(x, np.min(all_data, axis=0), np.max(all_data, axis=0), alpha=0.3, edgecolor=colors[model], facecolor=colors[model])

    ax.set_ylabel(key, weight='bold', fontsize=45, labelpad=10)
    ax.set_xlabel('Training Episodes')
    ax.legend()

    fig.tight_layout()
    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    else:
        image_suffix = key.replace('/', '__')
        fig.savefig(f'./results_mpsc/{image_suffix}.png', dpi=300)
    plt.close()


def benchmark_plot(system, task, algo):
    all_results = load_all_models(system, task, algo)
    X_GOAL = all_results['mpsf']['X_GOAL']

    uncert = all_results['none']['uncert_results']
    mpsf = all_results['mpsf']['cert_results']
    none = all_results['none']['cert_results']
    mpc = all_results['mpc']['cert_results']

    for i in [0]:
        print('Uncertified PPO')
        met.data = uncert
        print('num_violations:', calculate_state_violations(uncert, i))
        print('exp_return:', np.asarray(met.get_episode_returns())[i])
        print('exp_rmse:', np.asarray(met.get_episode_rmse())[i])

        print('\nCertified PPO')
        met.data = none
        print('num_violations:', calculate_state_violations(none, i))
        print('exp_return:', np.asarray(met.get_episode_returns())[i])
        print('exp_rmse:', np.asarray(met.get_episode_rmse())[i])

        print('\nCertified PPO trained with MPSC')
        met.data = mpsf
        print('num_violations:', calculate_state_violations(mpsf, i))
        print('exp_return:', np.asarray(met.get_episode_returns())[i])
        print('exp_rmse:', np.asarray(met.get_episode_rmse())[i])
        print('---------')

        print('\nNL-MPC')
        met.data = mpc
        print('num_violations:', calculate_state_violations(mpc, i))
        print('exp_return:', np.asarray(met.get_episode_returns())[i])
        print('exp_rmse:', np.asarray(met.get_episode_rmse())[i])
        print('---------')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(uncert['state'][i][:, 0], uncert['state'][i][:, 2], label='Uncertified', color='red')
        ax.plot(none['state'][i][:, 0], none['state'][i][:, 2], label='Certified (Std.)', color='cornflowerblue')
        ax.plot(mpsf['state'][i][:, 0], mpsf['state'][i][:, 2], label='Certified (Ours)', color='forestgreen')
        ax.plot(mpc['state'][i][:, 0], mpc['state'][i][:, 2], label='MPC', color='plum')
        ax.plot(X_GOAL[:, 0], X_GOAL[:, 2], label='Reference', color='black', linestyle='dashdot')
        rec1 = plt.Rectangle((0.9, 0), 2, 2, color='#f1d6d6')
        rec2 = plt.Rectangle((-1.9, 0), 1, 2, color='#f1d6d6')
        ax.add_patch(rec1)
        ax.add_patch(rec2)
        rec3 = plt.Rectangle((-0.9, 1.45), 0.975 * 2, 2, color='#f1d6d6')
        rec4 = plt.Rectangle((-0.9, -0.45), 0.975 * 2, 1, color='#f1d6d6')
        ax.add_patch(rec3)
        ax.add_patch(rec4)
        plt.xlim(-1.1, 1.1)
        plt.ylim(0.45, 1.55)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.legend()
        plt.show()


def calculate_state_violations(data, i):
    states = data['state'][i]
    num_viols = np.sum(np.any(states[:, [0, 2]] > [0.9, 1.45], axis=1) | np.any(states[:, [0, 2]] < [-0.9, 0.55], axis=1))
    return num_viols


if __name__ == '__main__':
    ordered_models = ['none', 'mpsf', 'mpc']

    colors = {
        'mpsf': 'royalblue',
        'none': 'plum',
    }

    def extract_rate_of_change_of_inputs(results_data, certified=True):
        return extract_rate_of_change(results_data, certified, order=1, mode='input')

    def extract_roc_cert(results_data, certified=True):
        return extract_rate_of_change_of_inputs(results_data, certified)

    def extract_roc_uncert(results_data, certified=False):
        return extract_rate_of_change_of_inputs(results_data, certified)

    def extract_rmse_cert(results_data, certified=True):
        return extract_rmse(results_data, certified)

    def extract_rmse_uncert(results_data, certified=False):
        return extract_rmse(results_data, certified)

    def extract_constraint_violations_cert(results_data, certified=True):
        return extract_constraint_violations(results_data, certified)

    def extract_constraint_violations_uncert(results_data, certified=False):
        return extract_constraint_violations(results_data, certified)

    def extract_reward_cert(results_data, certified=True):
        return extract_reward(results_data, certified)

    def extract_reward_uncert(results_data, certified=False):
        return extract_reward(results_data, certified)

    def extract_failed_cert(results_data, certified=True):
        return extract_failed(results_data, certified)

    def extract_failed_uncert(results_data, certified=False):
        return extract_failed(results_data, certified)

    def extract_length_cert(results_data, certified=True):
        return extract_length(results_data, certified)

    def extract_length_uncert(results_data, certified=False):
        return extract_length(results_data, certified)

    system_name = 'quadrotor_2D_attitude'
    task_name = 'track'
    algo_name = 'ppo'
    if len(sys.argv) == 4:
        system_name = sys.argv[1]
        task_name = sys.argv[2]
        algo_name = sys.argv[3]

    benchmark_plot(system_name, task_name, algo_name)
    # plot_all_logs(system_name, task_name, algo_name)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_magnitude_of_corrections)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_max_correction)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_roc_cert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_roc_uncert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_rmse_cert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_rmse_uncert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_constraint_violations_cert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_constraint_violations_uncert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_number_of_corrections)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_length_cert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_length_uncert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_reward_cert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_reward_uncert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_failed_cert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_failed_uncert)
    # plot_model_comparisons(system_name, task_name, algo_name, extract_feasible_iterations)
