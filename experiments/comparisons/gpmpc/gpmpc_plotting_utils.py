import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
import re
from datetime import datetime

from safe_control_gym.utils.utils import mkdirs
from safe_control_gym.utils.plotting import plot_from_logs, window_func, load_from_log_file, COLORS, LINE_STYLES
from safe_control_gym.controllers.mpc.mpc_utils import compute_state_rmse

# cartpole controller time interval 
dt = 1.0 / 15.0

def get_cost(test_runs):
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            cost = np.sum(test_runs[epoch][episode]['obs'] ** 2)
            costs[epoch, episode] = cost
    mean_cost = np.mean(costs, axis=1)
    return mean_cost

def get_reward(test_runs):
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    rewards = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            reward = np.sum(np.exp(2*test_runs[epoch][episode]['reward']))
            rewards[epoch, episode] = reward
    mean_reward = np.mean(rewards, axis=1)
    return mean_reward 

def get_average_rmse_error(runs):
    num_epochs = len(runs)
    num_episodes = len(runs[0])
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            mse, rmse = runs[epoch][episode]['total_rmse_state_error']
            costs[epoch, episode] = rmse

    mean_cost = np.mean(costs, axis=1)
    return mean_cost

def save_rmse_statistics(runs, num_points_per_epoch, dir):
    header = 'Train steps'
    num_epochs = len(runs)
    num_episodes = len(runs[0])
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            mse, rmse = runs[epoch][episode]['total_rmse_state_error']
            costs[epoch, episode] = rmse

    for episode in range(num_episodes):
        header = header + ',ep{}'.format(episode)

    samples = num_points_per_epoch
    rewards = np.array(costs)
    data = np.vstack((samples, rewards.T)).T
    fname = os.path.join(dir, 'rmse_stats.csv')
    
    np.savetxt(fname,
               data,
               delimiter=',',
               header=header)

    return costs

def plot_runs(all_runs, num_epochs, episode=0, ind=0, ylabel='x position', dir=None):
    plt.plot(all_runs[0][episode]['state'][:, ind], label='Linear MPC')
    for epoch in range(1, num_epochs):
        #plot the first episode of each epoch
        plt.plot(all_runs[epoch][episode]['state'][:, ind], label='GP-MPC %s' % epoch)
    plt.title(ylabel)
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.legend()
    save_str = 'ep%s_ind%s_state.png' % (episode, ind)
    if dir is not None:
        save_str = os.path.join(dir, save_str)
        plt.savefig(save_str)
    else:
        plt.show()
    plt.cla()
    plt.clf()



def plot_learning_curve(avg_rewards, num_points_per_epoch, stem, dir):
    samples = num_points_per_epoch
    rewards = np.array(avg_rewards)
    plt.plot(samples, rewards)
    plt.title('Avg Episode' + stem)
    plt.xlabel('Training Steps')
    plt.ylabel(stem)
    save_str = os.path.join(dir, stem + '.png')
    plt.savefig(save_str)
    plt.cla()
    plt.clf()
    data = np.vstack((samples, rewards)).T
    fname = os.path.join(dir, stem + '.csv')
    header = 'Train steps,Cost'
    np.savetxt(fname,
               data,
               delimiter=',',
               header=header)

def load_stats(legend_dir_specs, 
               scalar_names=[], 
               x_rescale_factor=None):
    """Gets all processed statistics for multiple scalars."""
    scalar_stats = {}
    for scalar_name in scalar_names:
        # Get all stats.
        stats = defaultdict(list)
        for l, dirs in legend_dir_specs.items():
            for d in dirs:
                # Pick from either log source (tensorboard or log text files).
                path = os.path.join(d, scalar_name + ".csv")
                # load from csv file
                data = np.genfromtxt(path, delimiter=',')
                x, y = data[:,0], data[:,1:]
                # _, x, _, y = load_from_log_file(path)
                x, y = np.asarray(x), np.asarray(y).squeeze()
                # rescale the x-axis, e.g. convert to training step to time 
                if x_rescale_factor is not None:
                    x *= x_rescale_factor

                stats[l].append([x, y])
        # Post-processing.
        x_max = float("inf")
        for _, runs in stats.items():
            for x, y in runs:
                # Align length of x data (get min across all runs & all algos).
                x_max = min(x_max, len(x))
        processed_stats = {}
        for name, runs in stats.items():
            # Use same x for all runs to an algo.
            # x = np.array([x[:x_max] for x, _ in runs])[0]
            x = np.array([x[:len(x)] for x, _ in runs])[0]
            # Different y for different runs.
            # y = np.stack([y[:x_max] for _, y in runs])
            y = np.stack([y[:len(_)] for _, y in runs])
            # Record stats.
            processed_stats[name] = {
                "x": x,
                "y": y,
                "mean": np.mean(y, axis=0),
                "std": np.std(y, axis=0),
                "median": np.median(y, axis=0),
                "quantiles": np.quantile(y, [0.25, 0.75], axis=0),
                "min": np.min(y, axis=0),
                "max": np.max(y, axis=0),
            }
        # Copy over stats.
        scalar_stats[scalar_name] = deepcopy(processed_stats)
    return scalar_stats

def get_last_stats(scalar_stats):
    """Gets the last-step stats for each category."""

    for scalar_name in scalar_stats.keys(): 
        processed_stats = scalar_stats[scalar_name]
        x_cat = []
        last_step_stats = []
        try:
            sample = list(processed_stats.keys())[0]
            if '.' in sample.split(".") or float(sample) < 1: # put float(sample) < 1 is for the case of 1E-4 for example.
                for _, name in enumerate(sorted(processed_stats.keys(), key=float)):
                    x_cat.append("{:.5f}".format(float(name)))
                    last_step_stats.append(processed_stats[name]["y"][:, -1])
            else:
                for _, name in enumerate(sorted(processed_stats.keys(), key=int)):
                    x_cat.append(name)
                    last_step_stats.append(processed_stats[name]["y"][:, -1])
        except:
            for _, name in enumerate(sorted(processed_stats.keys())):
                x_cat.append(name)
                last_step_stats.append(processed_stats[name]["y"][:, -1])
        

    return x_cat, last_step_stats

def plot_boxplot_from_stats(stats,
                            cats,
                            x_cat,
                            out_path,
                            file_name="HPO_comparison_same_scale.jpg"):
    """Plots boxplot from stats.

        args:
            stats = [[x][y][z]; [x][y][z]]
            cats = [[cat1][cat2][cat3]; [cat1][cat2][cat3]]
            x_cats = [[strategy1][strategy2]]
    
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 4))
    num_of_strategies = len(stats)
    num_of_subcats = len(stats[0])
    total_lens = 0
    for i in range(num_of_strategies):
        total_lens += len(stats[i])
        assert num_of_subcats == len(stats[i]), "Number of subcategories must be the same."

    stats_raw = np.array(stats[0]).T
    for i in range(1, num_of_strategies):
        stat_raw = np.array(stats[i]).T
        stats_raw = np.concatenate((stats_raw, stat_raw), axis=1)
    rearrange_index = np.arange(total_lens).reshape(num_of_strategies, -1).T.reshape(-1)

    arranged_stats_raw = stats_raw[:, rearrange_index]
    arranged_cats = np.array(cats).reshape(1, -1)[:, rearrange_index]

    bplot = ax.boxplot(arranged_stats_raw, vert=True, labels=arranged_cats.reshape(-1))
    box_colors = COLORS[:num_of_strategies]
    for i in range(total_lens):
        box = bplot['boxes'][i]
        box_x = []
        box_y = []
        for j in range(num_of_subcats):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax.add_patch(Polygon(box_coords, facecolor=box_colors[i % num_of_strategies]))

    ax.set_title('HPO Comparison')
    
    ax.set_yscale('log')
    ax.set_xlabel('Startegy')
    ax.set_ylabel('Cost')
    ax.set_ylim([0, 1])

    # Finally, add a basic legend
    for i in range(num_of_strategies):
        fig.text(0.91, 0.045+0.05*i, x_cat[i],
                backgroundcolor=box_colors[i], color='white', weight='roman',
                size='x-small')
    
    # auto adjust the figure size
    # fig.tight_layout()
    plt.savefig(os.path.join(out_path, file_name))

def plot_hpo_effort(config):
    """Gets the wall clock time and agent runs during hpo."""
    SAMPLER = "TPESampler" # "RandomSampler" or "TPESampler"
    SYS = "cartpole" # "cartpole" or "quadrotor_2D"
    hpo_folder = f'./experiments/comparisons/gpmpc/hpo/hpo_strategy_study_{SAMPLER}_{SYS}'
    hpo_strategy_runs = os.listdir(hpo_folder)

    # read std_out.txt to get total agent runs and duration time
    data_time = {}
    data_runs = {}
    for s in hpo_strategy_runs:
        parallel_job_folders = os.listdir(os.path.join(hpo_folder, s))
        duration_time = 0
        total_runs = 0
        for job_folder in parallel_job_folders:
            # check the folder is not ended with .sql
            if not job_folder.endswith('.sql'):
                with open(os.path.join(hpo_folder, s, job_folder, 'std_out.txt'), 'r') as file:
                    first_line = file.readline()
                    last_line = file.readlines()[-1]
                    
                    first_timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', first_line)
                    last_timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', last_line)
                    total_runs_match = re.search(r'Total runs: \d+', last_line)

                    first_timestamp = first_timestamp_match.group(0)
                    last_timestamp = last_timestamp_match.group(0)
                    total_runs = int(total_runs_match.group(0).split(': ')[1])

                    # Convert timestamps to datetime objects
                    start = datetime.strptime(first_timestamp, '%Y-%m-%d %H:%M:%S,%f')
                    end = datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S,%f')

                    # Calculate the duration time in hours
                    duration_hours = (end - start).total_seconds() / 3600
                    

                    # check if duration time is larger
                    if duration_time < duration_hours:
                        duration_time = duration_hours
                    total_runs += int(total_runs_match.group(0).split(': ')[1])

                data_time[s] = {'Duration Time (hours)': duration_time}
                data_runs[s] = {'Total Runs': total_runs}
        
    # add to pandas dataframe
    df = pd.DataFrame(data_time)

    melted_df = pd.melt(df, var_name='Category_Run', value_name='Duration Time (hours)')
    melted_df['Category'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[1])
    melted_df['Run'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[0])
    melted_df.sort_values(by=['Category'])

    plt.figure(figsize=(10, 6))
    # sns.barplot(x='Category', y='Duration Time (hours)', hue='Run', data=melted_df)
    # plt.legend(title='Run')
    sns.barplot(x='Category', y='Duration Time (hours)', data=melted_df, order=['s1', 's2', 's3', 's4', 's5'])
    plt.xlabel('Category')
    plt.ylabel('Duration Time (hours)')
    plt.title('HPO Strategy Effort')
    plt.show()
    plt.savefig(os.path.join(config.plot_dir, "HPO_time_comparison.jpg"))
    plt.close()
    
    # pickle melted_df
    melted_df.to_pickle(os.path.join(config.plot_dir, "gpmpc_hpo_effort.pkl"))

    # add to pandas dataframe
    df = pd.DataFrame(data_runs)

    melted_df = pd.melt(df, var_name='Category_Run', value_name='Total Runs')
    melted_df['Category'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[1])
    melted_df['Run'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[0])

    plt.figure(figsize=(10, 6))
    # sns.barplot(x='Category', y='Total Runs', hue='Run', data=melted_df)
    # plt.legend(title='Run')
    sns.barplot(x='Category', y='Total Runs', data=melted_df, order=['s1', 's2', 's3', 's4', 's5'])
    plt.xlabel('Category')
    plt.ylabel('Total Agent Runs')
    plt.yscale('log')
    plt.title('HPO Strategy Effort')
    plt.show()
    plt.savefig(os.path.join(config.plot_dir, "HPO_agent_runs_comparison.jpg"))
    plt.close()

    print("Hyperparameter optimization effort plotting done.")

def plot_hpo_eval(config):
    """Gets the plot and csv for performance (in RMSE)."""
    SAMPLER = "TPESampler" # "RandomSampler" or "TPESampler"
    SYS = "cartpole" # "cartpole" or "quadrotor_2D"
    visualize_hp_vectors = False
    legend_map_s1 = {
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run1_s1": "run1_s1",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run2_s1": "run2_s1",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run3_s1": "run3_s1"
    }
    legend_map_s2 = {
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run1_s2": "run1_s2",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run2_s2": "run2_s2",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run3_s2": "run3_s2",
    }
    legend_map_s3 = {
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run1_s3": "run1_s3",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run2_s3": "run2_s3",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run3_s3": "run3_s3",
    }
    legend_map_s4 = {
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run1_s4": "run1_s4",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run2_s4": "run2_s4",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run3_s4": "run3_s4",
    }
    legend_map_s5 = {
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run1_s5": "run1_s5",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run2_s5": "run2_s5",
        f"hpo_strategy_study_{SAMPLER}_{SYS}/run3_s5": "run3_s5",
    }
    algo_name_map_s1 = {
        "run1_s1": "run1_s1",
        "run2_s1": "run2_s1",
        "run3_s1": "run3_s1",
    }
    algo_name_map_s2 = {
        "run1_s2": "run1_s2",
        "run2_s2": "run2_s2",
        "run3_s2": "run3_s2",
    }
    algo_name_map_s3 = {
        "run1_s3": "run1_s3",
        "run2_s3": "run2_s3",
        "run3_s3": "run3_s3",
    }
    algo_name_map_s4 = {
        "run1_s4": "run1_s4",
        "run2_s4": "run2_s4",
        "run3_s4": "run3_s4",
    }
    algo_name_map_s5 = {
        "run1_s5": "run1_s5",
        "run2_s5": "run2_s5",
        "run3_s5": "run3_s5",
    }
    scalar_name_map = {
        # "figs/avg_rmse_cost_learning_curve": "Cost",
        "figs/avg_reward_learning_curve": "Reward",
    }

    legend_map_list = [legend_map_s1, legend_map_s2, legend_map_s3, legend_map_s4, legend_map_s5]
    algo_name_map_list = [algo_name_map_s1, algo_name_map_s2, algo_name_map_s3, algo_name_map_s4, algo_name_map_s5]

    data = {}
    for index, (legend_map, algo_name_map) in enumerate(zip(legend_map_list, algo_name_map_list)):
        # Collect results.
        spec = {}
        for d, legend in legend_map.items():
            seed_dirs = os.listdir(os.path.join(config.plot_dir, d))
            spec[legend] = [os.path.join(config.plot_dir, d, sd) for sd in seed_dirs]

        # Get all stats.
        scalar_stats = load_stats(spec, 
                                scalar_names=scalar_name_map, 
                                x_rescale_factor=dt) 

        # Get last step stats
        x_cat, last_step_stats = get_last_stats(scalar_stats)
        for i in range(len(x_cat)):
            # data[x_cat[i]] = np.array([last_step_stats[i].mean()])
            data[x_cat[i]] = last_step_stats[i]

    # hpo evaluation
    df = pd.DataFrame(data)
    melted_df = pd.melt(df, var_name='Category_Run', value_name='Reward')

    melted_df['Category'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[1])
    melted_df['Run'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[0])

    # print the statistics of each category
    print(melted_df.groupby(['Category_Run']).describe())
    print(melted_df.groupby(['Category']).describe())

    plt.figure(figsize=(10, 6))
    # sns.boxplot(x='Category', y='RMSE Cost', hue='Run', data=melted_df)
    # plt.legend(title='Run')
    sns.boxenplot(x='Category', y='Reward', data=melted_df, k_depth="proportion")
    plt.xlabel('Category')
    plt.ylabel('Reward')
    # plt.yscale('log')
    plt.title('HPO Strategy Evaluation')
    plt.show()
    plt.savefig(os.path.join(config.plot_dir, f"HPO_comparison_{SAMPLER}.jpg"))
    plt.close()

    # pickle melted_df
    melted_df.to_pickle(os.path.join(config.plot_dir, "gpmpc_hpo_eval.pkl"))
    
    print("HPO evaluation plotting done.")

def filter_sequences(x_seq, actions, x_next_seq, threshold):
    # Find where the difference in step is greater than the filter threshold
    x_diff_abs = np.abs(x_next_seq - x_seq)
    rows_to_keep = np.all(x_diff_abs<1, axis=1)
    x_seq_filt = x_seq[rows_to_keep, :]
    x_next_seq_filt = x_next_seq[rows_to_keep, :]
    actions_filt = actions[rows_to_keep, :]
    return x_seq_filt, actions_filt, x_next_seq_filt

def get_quad_cost(test_runs, ref):
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            cost = np.sum((test_runs[epoch][episode]['obs'][1:,[0,2]] - ref[:,[0,2]]) ** 2)
            costs[epoch, episode] = cost
    mean_cost = np.mean(costs, axis=1)
    return mean_cost


def get_quad_average_rmse_error(runs, ref):
    num_epochs = len(runs)
    num_episodes = len(runs[0])
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            mse, rmse = runs[epoch][episode]['total_rmse_state_error']
            costs[epoch, episode] = rmse

    mean_cost = np.mean(costs, axis=1)
    return mean_cost

def get_quad_average_rmse_error_xz_only(runs, ref):
    num_epochs = len(runs)
    num_episodes = len(runs[0])
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            mse, rmse = compute_state_rmse(runs[epoch][episode]['state'][1:,[0,2]] - ref[:,[0,2]])
            costs[epoch, episode] = rmse

    mean_cost = np.mean(costs, axis=1)
    return mean_cost

def make_plots(test_runs, train_runs, num_inds, dir):
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    fig_dir = os.path.join(dir,'figs')
    mkdirs(fig_dir)

    # Make plot of all trajectories.
    num_points_per_epoch = []
    for episode_i in range(num_episodes):
        for ind in range(num_inds):
            ylabel = 'x%s' % ind
            plot_runs(test_runs, num_epochs, episode=episode_i, ind=ind, ylabel=ylabel, dir=fig_dir)

    # Compute the number of training points (x-axis for most figures).
    num_points = 0
    num_points_per_epoch.append(num_points)
    for epoch in range(1, num_epochs):
        num_train_episodes = len(train_runs[epoch])
        for episode in range(num_train_episodes):
            num_points += train_runs[epoch][episode]['obs'].shape[0]
        num_points_per_epoch.append(num_points)


    # Plot violation data
    nsamp, viol, mean_viol, maximums = get_constraint_violations(test_runs,
                                                                 train_runs)
    plot_constraint_violation(num_points_per_epoch, viol, fig_dir)
    avg_maxs = get_avg_of_max_viol_theta_dot(maximums)
    plot_average_peak_theta_dot_viol(num_points_per_epoch, avg_maxs, fig_dir)

    # Plot Learning Curves
    avg_rmse_error = get_average_rmse_error(test_runs)
    rmse_stats = save_rmse_statistics(test_runs, num_points_per_epoch, fig_dir)
    plot_learning_curve(avg_rmse_error, num_points_per_epoch, 'avg_rmse_cost_learning_curve', fig_dir)
    common_costs = get_cost(test_runs)
    plot_learning_curve(common_costs, num_points_per_epoch, 'common_cost_learning_curve', fig_dir)

    # Plot average reward
    avg_reward = get_reward(test_runs)
    plot_learning_curve(avg_reward, num_points_per_epoch, 'avg_reward_learning_curve', fig_dir)


def make_quad_plots(test_runs, train_runs, trajectory, num_inds, dir):
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    fig_dir = os.path.join(dir,'figs')
    mkdirs(fig_dir)
    num_points_per_epoch = []
    for episode_i in range(num_episodes):
        for ind in range(num_inds):
            ylabel = 'x%s' % ind

            plot_runs(test_runs, num_epochs, episode=episode_i, ind=ind, ylabel=ylabel, dir=fig_dir)
    num_points = 0
    num_points_per_epoch.append(num_points)
    for epoch in range(1, num_epochs):
        num_train_episodes = len(train_runs[epoch])
        for episode in range(num_train_episodes):
            num_points += train_runs[epoch][episode]['obs'].shape[0]
        num_points_per_epoch.append(num_points)

    common_costs = get_quad_cost(test_runs, trajectory)
    plot_learning_curve(common_costs, num_points_per_epoch, 'common_xz_cost_learning_curve', fig_dir)
    rmse_error = get_quad_average_rmse_error(test_runs, trajectory)
    plot_learning_curve(rmse_error, num_points_per_epoch, 'rmse_error_learning_curve', fig_dir)
    rmse_error_xz = get_quad_average_rmse_error_xz_only(test_runs, trajectory)
    plot_learning_curve(rmse_error_xz, num_points_per_epoch, 'rmse_xz_error_learning_curve', fig_dir)

def gather_training_samples(all_runs, epoch_i, num_samples, rand_generator=None):
    n_episodes = len(all_runs[epoch_i].keys())
    num_samples_per_episode = int(num_samples/n_episodes)
    x_seq_int = []
    x_next_seq_int = []
    actions_int = []
    for episode_i in range(n_episodes):
        run_results_int = all_runs[epoch_i][episode_i]
        n = run_results_int['action'].shape[0]
        if num_samples_per_episode < n:
            if rand_generator is not None:
                rand_inds_int = rand_generator.choice(n-1, num_samples_per_episode, replace=False)
            else:
                rand_inds_int = np.arange(num_samples_per_episode)
        else:
            rand_inds_int = np.arange(n-1)
        next_inds_int = rand_inds_int + 1
        x_seq_int.append(run_results_int.obs[rand_inds_int, :])
        actions_int.append(run_results_int.action[rand_inds_int, :])
        x_next_seq_int.append(run_results_int.obs[next_inds_int, :])
    x_seq_int = np.vstack(x_seq_int)
    actions_int = np.vstack(actions_int)
    x_next_seq_int = np.vstack(x_next_seq_int)

    return x_seq_int, actions_int, x_next_seq_int


def gather_training_samples_from_all_data(all_runs, num_samples):
    n_epochs = len(all_runs.keys())
    n_episodes_per_epoch = len(all_runs[0].keys())
    num_samples_per_episode = int(num_samples/(n_episodes_per_epoch*n_epochs))
    x_seq_int = []
    x_next_seq_int = []
    actions_int = []
    for epoch_i in range(n_epochs):
        for episode_i in range(n_episodes_per_epoch):
            run_results_int = all_runs[epoch_i][episode_i]
            n = run_results_int['action'].shape[0]
            if num_samples_per_episode < n:
                #rand_inds_int = np.random.choice(n-1, num_samples_per_episode)
                rand_inds_int = np.arange(num_samples_per_episode)
            else:
                rand_inds_int = np.arange(n-1)
            next_inds_int = rand_inds_int + 1
            x_seq_int.append(run_results_int.obs[rand_inds_int, :])
            actions_int.append(run_results_int.action[rand_inds_int, :])
            x_next_seq_int.append(run_results_int.obs[next_inds_int, :])
    x_seq_int = np.vstack(x_seq_int)
    actions_int = np.vstack(actions_int)
    x_next_seq_int = np.vstack(x_next_seq_int)

    return x_seq_int, actions_int, x_next_seq_int


def make_traking_plot(runs, traj, dir, impossible=True):
    num_epochs = len(runs.keys())
    plt.figure()
    plt.plot(runs[0][0]['obs'][:, 0], runs[0][0]['obs'][:, 2], label='Linear MPC')
    traj_lin = np.vstack((runs[0][0]['obs'][:, 0], runs[0][0]['obs'][:, 2])).T
    np.savetxt(os.path.join(dir, 'traj_lin_mpc.csv'), traj_lin, delimiter=',')
    for epoch in range(1, num_epochs):
        traj1 = np.vstack((runs[epoch][0]['obs'][:, 0], runs[epoch][0]['obs'][:, 2])).T
        np.savetxt(os.path.join(dir, 'traj_%s.csv' % epoch), traj1, delimiter=',')
        plt.plot(runs[epoch][0]['obs'][:, 0], runs[epoch][0]['obs'][:, 2], label='GP-MPC %s' % epoch)
    plt.plot(traj[:,0], traj[:,2], 'k',label='Reference')
    if impossible:
        plt.plot([-0.4,-0.4],[0.0, 0.9], 'r', label='Limit')
        plt.plot([0.4,0.4],[0.0, 0.9], 'r')
        plt.plot([-0.4,0.4],[0.9, 0.9], 'r')
        plt.plot([-0.4,0.4],[0.0, 0.0], 'r')
    plt.legend()
    if impossible:
        plt.title("Quadrotor Impossible Tracking")
    else:
        plt.title("Quadrotor Tracking")
    plt.xlabel('X position (m)')
    plt.ylabel('Z position (m)')
    save_str = os.path.join(dir, 'quad_traj.png')
    plt.savefig(save_str)

def get_constraint_violations(test_runs,
                              train_runs):
    num_train_samples_by_epoch = []
    violations_per_epoch = []
    max_violations_per_epoch = []
    mean_violations_per_epoch = []
    num_samples = 0
    num_epochs = len(train_runs.keys())
    n_train_samples_per_epoch = 0
    for epoch in range(num_epochs):
        violations_per_episode = []
        max_violations_per_episode = []
        num_train_samples_per_episode = len(train_runs[epoch].keys())
        for train_episode in range(num_train_samples_per_episode):
            n_train_samples_per_epoch += len(train_runs[epoch][train_episode]['info'])
        num_test_episodes_per_epoch = len(test_runs[epoch].keys())
        for test_episode in range(num_test_episodes_per_epoch):
            violations = 0
            max_violations = np.zeros(test_runs[epoch][test_episode]['info'][0]['constraint_values'].shape)
            n = len(test_runs[epoch][test_episode]['info'])
            for i in range(n):
                #violations += test_runs[epoch][test_episode]['info'][i]['constraint_violation'] # Due to bug.
                violations += int(np.any(test_runs[epoch][test_episode]['info'][i]['constraint_values'] > 0))
                max_violations = np.maximum(max_violations,
                                            test_runs[epoch][test_episode]['info'][i]['constraint_values'])

            violations_per_episode.append(violations)
            max_violations_per_episode.append(max_violations)
        num_train_samples_by_epoch.append(n_train_samples_per_epoch)
        violations_per_epoch.append(violations_per_episode)
        max_violations_per_epoch.append(np.vstack(max_violations_per_episode))
        mean_violations_per_epoch.append(np.mean(violations_per_episode))
        num_samples += n
    return num_train_samples_by_epoch, violations_per_epoch, mean_violations_per_epoch, max_violations_per_epoch

def plot_constraint_violation(viol_samp, viols, dir):
    violations = np.array(viols)
    train_time = np.array(viol_samp)
    mean_viol = np.mean(violations, axis=1)
    max = np.max(violations, axis=1)
    min = np.min(violations, axis=1)

    plt.plot(train_time, mean_viol, label='mean')
    plt.plot(train_time, max, label='max')
    plt.plot(train_time, min, label='min')
    plt.legend()
    plt.xlabel('Train Steps')
    plt.ylabel('Number of violations')
    stem = 'number_viol'
    fname = os.path.join(dir, stem + '.png')
    plt.savefig(fname)
    plt.cla()
    plt.clf()

    data = np.vstack((train_time, mean_viol, max, min)).T
    fname = os.path.join(dir, stem + '.csv')
    np.savetxt( fname,
                data,
                delimiter=',',
                header='Train Steps (s),Mean,Max,Min')

def get_avg_of_max_viol_theta_dot(maximums):
    """ get the average of the max violations in theta_dot across episodes for each epoch."""
    num_epochs = len(maximums)
    max_avgs = []
    for epoch in range(num_epochs):
        max_avgs.append(np.mean(np.max(maximums[epoch][:,[3,7]], axis=1)))
    return max_avgs

def plot_average_peak_theta_dot_viol(train_sample, avg_max, dir):
    plt.plot(train_sample, avg_max)
    plt.xlabel('Training Time (s)')
    plt.ylabel('Avg Peak Violation (rad/s)')
    plt.title('Theta_dot Average Peak Violation')
    stem = 'peak_viol'
    fname = os.path.join(dir, stem + '.png')
    plt.savefig(fname)
    plt.cla()
    plt.clf()
    data = np.vstack((train_sample, avg_max)).T
    fname = os.path.join(dir, stem + '.csv')
    np.savetxt(fname,
               data,
               delimiter=',',
               header='Train Time (s),Avg peak violation (rad/s)')

def plot_robustness(runs, pole_lengths, label, dir):
    num_coeff = len(runs)
    # compute common costs
    avg_costs = get_cost(runs)
    plt.plot(pole_lengths, avg_costs)
    plt.title('GP-MPC' + label + ' robustness')
    plt.xlabel(label + ' Bounds')
    plt.ylabel('Normalized Common Cost')
    plt.savefig(os.path.join(dir,'common_cost_robust_plot.png'))

    plt.cla()
    plt.clf()
    data = np.vstack((pole_lengths, avg_costs)).T
    fname = os.path.join(dir, 'common_cost_robust_plot.csv')
    header = 'Coeff,Avg Cost'
    np.savetxt(fname,
               data,
               delimiter=',',
               header=header)

def plot_robustness_runs(all_runs, num_epochs, parameters, episode=0, ind=0, ylabel='x position', dir=None):
    for epoch in range(0, num_epochs):
        # plot the first episode of each epoch
        plt.plot(all_runs[epoch][episode]['state'][:, ind], label='%s' % parameters[epoch])
    plt.title(ylabel)
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.legend()
    save_str = 'ep%s_ind%s_state.png' % (episode, ind)
    if dir is not None:
        save_str = os.path.join(dir, save_str)
        plt.savefig(save_str)
    else:
        plt.show()
    plt.cla()
    plt.clf()

def table_csv(runs, dir):
    num_epochs = len(runs)
    num_epiosdes = len(runs[0])
    rmse_errors = np.zeros((num_epiosdes, num_epochs))
    for epoch in range(num_epochs):
        for episodes in range(num_epiosdes):
            mse, rmse = runs[epoch][episodes]['total_rmse_state_error']
            rmse_errors[episodes, epoch] = rmse
    np.savetxt(os.path.join(dir, 'total_rmse_state_error_table.csv'), rmse_errors, delimiter=',')


def plot_robustness_rmse(runs, pole_lengths, label, dir):
    num_coeff = len(runs)
    # compute common costs
    avg_costs = get_average_rmse_error(runs)
    plt.plot(pole_lengths, avg_costs)
    plt.title('GP-MPC ' + label+ ' Robustness')
    plt.xlabel(label + ' Bounds')
    plt.ylabel('Average RMSE')
    plt.savefig(os.path.join(dir,'rmse_robust_plot.png'))

    plt.cla()
    plt.clf()
    data = np.vstack((pole_lengths, avg_costs)).T
    fname = os.path.join(dir, 'rmse_robust_plot.csv')
    header = 'Coeff,Avg Cost'
    np.savetxt(fname,
               data,
               delimiter=',',
               header=header)


def plot_all_robustness_runs(runs, parameters, dir):
    num_epochs = len(runs)
    num_episodes = len(runs[0])
    fig_dir = os.path.join(dir, 'figs')
    mkdirs(fig_dir)
    num_inds = runs[0][0]['state'].shape[1]

    # Make plot of all trajectories.
    num_points_per_epoch = []
    for episode_i in range(num_episodes):
        for ind in range(num_inds):
            ylabel = 'x%s' % ind
            plot_robustness_runs(runs, num_epochs, parameters, episode=episode_i, ind=ind, ylabel=ylabel, dir=fig_dir)

def plot_constraint_from_csv(fname,
                             plot_name):
    data = np.genfromtxt(fname, delimiter=',')
    plt.plot(data[:,0], data[:,1])
    plt.title(plot_name)
    plt.xlabel('Train Steps (s)')
    plt.ylabel('Avg number of violations')
    plt.show()

def plot_data_eff_from_csv(fname,
                             plot_name):
    data = np.genfromtxt(fname, delimiter=',')
    plt.plot(data[:,0], data[:,1])
    plt.title(plot_name)
    plt.xlabel('Train Steps (s)')
    plt.ylabel('Eval. Cost')
    plt.show()

def plot_robustness_from_csv(fname,
                             plot_name,
                             x_label):
    data = np.genfromtxt(fname, delimiter=',')
    plt.plot(data[:,0], data[:,1])
    plt.title(plot_name)
    plt.xlabel(x_label)
    plt.ylabel('Eval. Cost')
    plt.show()

def plot_impossible_traj_from_csv(fnames):
    plt.figure()
    n = len(fnames)
    lin_mpc_data = np.genfromtxt(fnames[0], delimiter=',')
    plt.plot(lin_mpc_data[:,0], lin_mpc_data[:,1], label='Linear MPC')
    for i in range(1, n):
        traj = np.genfromtxt(fnames[i], delimiter=',')
        plt.plot(traj[:,0], traj[:,1], label='GP-MPC %s' % i)
    t = np.linspace(0, 2*np.pi, num=len(traj))
    trajx = 0.5*np.sin(t)
    trajy = 0.5*np.cos(t) + 0.5
    plt.plot(trajx, trajy, 'k',label='Reference')
    plt.plot([-0.4,-0.4],[0.0, 0.9], 'r', label='Limit')
    plt.plot([0.4,0.4],[0.0, 0.9], 'r')
    plt.plot([-0.4,0.4],[0.9, 0.9], 'r')
    plt.plot([-0.4,0.4],[0.0, 0.0], 'r')
    plt.legend()
    plt.title("Quadrotor Impossible Tracking")
    plt.xlabel('X position (m)')
    plt.ylabel('Z position (m)')
    plt.show()

def plot_ctrl_perf(fname, labels):
    data = np.genfromtxt(fname, delimiter=',')
    n = data.shape[1]
    fig, ax = plt.subplots(n-1,1)
    for i in range(1,n):
        ax[i-1].plot(data[:,0], data[:,i])
        ax[i-1].set_xlabel('Time (s)')
        ax[i-1].set_ylabel(labels[i-1], fontsize=6)
    plt.show()
