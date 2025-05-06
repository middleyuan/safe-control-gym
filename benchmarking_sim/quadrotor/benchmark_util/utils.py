import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

from benchmarking_sim.quadrotor.mb_experiment_rollout import run

def load_gym_data(data_dir):
    traj_data = np.load(data_dir, allow_pickle=True)
    obs = traj_data['trajs_data']['obs'][0]
    state = traj_data['trajs_data']['state'][0]
    act = traj_data['trajs_data']['action'][0]
    rew = traj_data['trajs_data']['reward'][0]
    ref = traj_data['trajs_data']['info'][0][0]['x_reference']
    error = []
    for i in range(1, len(traj_data['trajs_data']['info'][0])):
        error.append(np.sqrt(traj_data['trajs_data']['info'][0][i]['mse']))
    error = np.array(error)
    # = traj_data['trajs_data']['info'][0][0]['error']
    rmse = traj_data['metrics']['rmse']
    results = {'obs': obs, 
               'state': state, 
               'action': act, 
               'rew': rew,
               'ref': ref,
               'rmse': rmse,
               'error': error,
               }
    return results

def extract_rollouts(notebook_dir, data_folder, controller_name, additional=''):
    # print('notebook_dir', notebook_dir)
    data_folder_path = os.path.join(notebook_dir, controller_name, data_folder)
    # print('data_folder_path', data_folder_path)
    assert os.path.exists(data_folder_path), 'data_folder_path does not exist'

    # find all the subfolders in the data_folder_path
    subfolders = [f.path for f in os.scandir(data_folder_path) if f.is_dir()]
    # print('subfolders', subfolders)
    # load the row 'rmse in the metrics.txt
    metrics = []
    traj_resutls = []
    timing_data = []
    for subfolder in subfolders:
        file_path = os.path.join(subfolder, 'metrics.txt')
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if not line.startswith('rmse_std') and line.startswith('rmse'):
                    # split the text between : and \n
                    line = line.split(': ')[-1].split('\n')[0]
                    metrics.append(eval(line))
                if line.startswith('avarage_inference_time'):
                    line = line.split(': ')[-1].split('\n')[0]
                    timing_data.append(eval(line))

        # find the file ends with pickle and get the data
        for file in os.listdir(subfolder):
            if file.endswith('.pkl'):
                file_path = os.path.join(subfolder, file)
                # print('file_path', file_path)
                results = np.load(file_path, allow_pickle=True)
                traj_data = results['trajs_data']['obs'][0]
                traj_resutls.append(traj_data)

    traj_resutls = np.array(traj_resutls)
    traj_file_name = f'traj_results_{controller_name}{additional}.npy'
    np.save(traj_file_name, traj_resutls)
    print('traj_results.shape', traj_resutls.shape)
    # print('metrics', metrics)
    rmse_mean_mpc = np.mean(metrics)
    rmse_std_mpc = np.std(metrics)
    print(f'rmse_{controller_name}{additional}', rmse_mean_mpc, rmse_std_mpc)
    return traj_resutls, metrics, timing_data

def run_rollouts(task_description):

    additional = getattr(task_description, 'additional', '')
    start_seed = getattr(task_description, 'start_seed', 1)
    num_seed = getattr(task_description, 'num_seed', 10)
    algo = getattr(task_description, 'algo', 'pid')
    num_runs_per_seed = getattr(task_description, 'num_runs_per_seed', 1)
    SYS = getattr(task_description, 'SYS', 'quadrotor_2D_attitude')
    noise_factor = getattr(task_description, 'noise_factor', 1)
    eval_task = getattr(task_description, 'eval_task', None)
    dw_height = getattr(task_description, 'dw_height', None)
    dw_height_scale = getattr(task_description, 'dw_height_scale', None)
    gp_model_tag = getattr(task_description, 'gp_model_tag', '')
    ctrl_tag = getattr(task_description, 'ctrl_tag', '')
    
    for seed in range(start_seed, num_seed + start_seed):
        run(n_episodes=num_runs_per_seed,
            seed=seed, 
            Additional=additional, 
            ALGO=algo,
            SYS=SYS,
            noise_factor=noise_factor,
            dw_height=dw_height,
            dw_height_scale=dw_height_scale,
            eval_task=eval_task,
            gp_model_tag=gp_model_tag,
            ctrl_tag=ctrl_tag,
            )

def plot_xz_trajectory_with_hull(ax, traj_data, label=None, 
                                 traj_color='skyblue', hull_color='lightblue',
                                 alpha=0.5, padding_factor=1.1):
    '''Plot trajectories with convex hull showing variance over seeds.
    
    Args:
        ax (Axes): Matplotlib axes.
        traj_data (np.ndarray): Trajectory data of shape (num_seeds, num_steps, 6).
        padding_factor (float): Padding factor for the convex hull.
    '''
    num_seeds, num_steps, _ = traj_data.shape

    print('traj data shape:', traj_data.shape)
    mean_traj = np.mean(traj_data, axis=0)
    
    ax.plot(mean_traj[:, 0], mean_traj[:, 2], color=traj_color, label=label)
    # plot the hull
    for i in range(num_steps - 1):
        # plot the hull at a single step
        points_at_step = traj_data[:, i, [0, 2]]
        hull = ConvexHull(points_at_step)
        cent = np.mean(points_at_step, axis=0) # center
        pts = points_at_step[hull.vertices] # vertices
        poly = Polygon(padding_factor*(pts - cent) + cent, 
                       closed=True,  
                       capstyle='round', 
                       facecolor=hull_color,
                       alpha=alpha)
        ax.add_patch(poly)

        # connecting consecutive convex hulls
        points_at_next_step = traj_data[:, i+1, [0, 2]]
        points_connecting = np.concatenate([points_at_step, points_at_next_step], axis=0)
        hull_connecting = ConvexHull(points_connecting)
        cent_connecting = np.mean(points_connecting, axis=0)
        pts_connecting = points_connecting[hull_connecting.vertices]
        poly_connecting = Polygon(padding_factor*(pts_connecting - cent_connecting) + cent_connecting, 
                                  closed=True,  
                                  capstyle='round', 
                                  facecolor=hull_color,
                                  alpha=alpha)
        ax.add_patch(poly_connecting)

def plot_trajectory(notebook_dir, data_folder, title):
    from safe_control_gym.utils.configuration import ConfigFactory
    from functools import partial
    from safe_control_gym.utils.registration import make
    #########################################################################
    # launch SCG to get reference trajectory X_GOAL
    ALGO = 'pid'
    SYS = 'quadrotor_2D_attitude'
    TASK = 'tracking'
    # PRIOR = '200_hpo'
    PRIOR = '100'
    agent = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' else SYS
    SAFETY_FILTER = None

    # check if the config file exists
    assert os.path.exists(f'./config_overrides/{SYS}_{TASK}.yaml'), f'../config_overrides/{SYS}_{TASK}.yaml does not exist'
    assert os.path.exists(f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), f'../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
    if SAFETY_FILTER is None:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', agent,
                        '--overrides',
                            f'./config_overrides/{SYS}_{TASK}.yaml',
                            f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                        '--seed', '2',
                        '--use_gpu', 'True',
                        '--output_dir', f'./{ALGO}/results',
                            ]
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    # merge config and create output directory
    config = fac.merge()
    # Create an environment
    env_func = partial(make,
                        config.task,
                        seed=config.seed,
                        **config.task_config
                        )
    random_env = env_func(gui=False)
    X_GOAL = random_env.X_GOAL
    ##########################################################################
    # load trajectory pkl files, load from folder
    controller_name = 'pid'
    fmpc_data_path = os.path.join(notebook_dir, controller_name, data_folder)
    assert os.path.exists(fmpc_data_path), 'data_folder_path does not exist'
    # fmpc_data_path = '/home/tobias/Studium/masterarbeit/code/safe-control-gym/benchmarking_sim/quadrotor/fmpc/results_rollout/temp'
    fmpc_data_dirs = [d for d in os.listdir(fmpc_data_path) if os.path.isdir(os.path.join(fmpc_data_path, d))]
    fmpc_traj_data_name = f'{controller_name}_data_quadrotor_traj_tracking.pkl'
    fmpc_traj_data_name = [os.path.join(d, fmpc_traj_data_name) for d in fmpc_data_dirs]

    fmpc_data = []
    for d in fmpc_traj_data_name:
        fmpc_data.append(np.load(os.path.join(fmpc_data_path, d), allow_pickle=True))
    fmpc_traj_data = [d['trajs_data']['obs'][0] for d in fmpc_data]
    fmpc_traj_data = np.array(fmpc_traj_data)
    print(fmpc_traj_data.shape) # seed, time_step, obs
    # take average of all seeds
    mpc_mean_traj_data = np.mean(fmpc_traj_data, axis=0)
    print(mpc_mean_traj_data.shape) # (mean_541, 6)


    # Define Colors
    ref_color = 'black'
    fmpc_color = 'purple'
    fmpc_hull_color = 'violet'

    # plot the state path x, z [0, 2]
    title_fontsize = 20
    legend_fontsize = 14
    axis_label_fontsize = 14
    axis_tick_fontsize = 12

    fig, ax = plt.subplots(figsize=(8, 4))
    # adjust the distance between title and the plot
    fig.subplots_adjust(top=0.2)
    ax.plot(X_GOAL[:, 0], X_GOAL[:, 2], color=ref_color, linestyle='-.', label='Reference')
    # ax.plot()
    ax.set_xlabel('$x$ [m]', fontsize=axis_label_fontsize)
    ax.set_ylabel('$z$ [m]', fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=axis_tick_fontsize)
    # ax.set_title('State path in $x$-$z$ plane')
    # set the super title
    # if not generalization:
    #     fig.suptitle(f'Evaluation ({plot_name})', fontsize=title_fontsize)
    # else:
    #     fig.suptitle(f'Generalization ({plot_name})', fontsize=title_fontsize)
    fig.suptitle(title, fontsize=title_fontsize)
    ax.set_ylim(0.35, 1.85)
    ax.set_xlim(-1.6, 1.6)
    fig.tight_layout()


    # plot the convex hull of each steps
    k = 1.1 # padding factor
    alpha = 0.2

    plot_xz_trajectory_with_hull(ax, fmpc_traj_data, label='FMPC',
                                    traj_color=fmpc_color, hull_color=fmpc_hull_color,
                                    alpha=alpha, padding_factor=k)

    ax.legend(ncol=5, loc='upper center', fontsize=legend_fontsize)

    fig.savefig(os.path.join(fmpc_data_path, 'xz_path_performance.png'), dpi=300, bbox_inches='tight')

