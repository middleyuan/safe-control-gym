import numpy as np
from pathlib import Path

from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

from benchmarking_sim.quadrotor.mb_experiment_rollout import run

plot_colors = {
    'GP-MPC': 'royalblue',
    'PPO': 'darkorange',
    'SAC': 'red',
    'DPPO': 'pink',
    'Geometric Control': 'darkgray',
    'Linear MPC': 'green',
    'Nonlinear MPC': 'cadetblue',
    'F-MPC': 'darkblue',
    'iLQR': 'slateblue',
    'LQR': 'blueviolet',
    'PPO-MPC': 'tan',
    'MAX': 'none',
    'MIN': 'none',
    'Reference': 'black',
    # 'Geometric Control': 'tab:gray',
}

tag_ctrl_list = {
    'iLQR': 'ilqr',
    'LQR': 'lqr',
    'Geometric Control': 'pid',
    'Linear MPC': 'linear_mpc_acados',
    'Nonlinear MPC': 'mpc_acados',
    'F-MPC': 'fmpc',
    'GP-MPC': 'gpmpc_acados_TP',
    'PPO': 'ppo',
    'SAC': 'sac',
    'DPPO': 'dppo',
    'PPO-MPC': 'ppo_mpc',
    'PPO-ID': 'ppo_id',
    'SAC-ID': 'sac_id',
    'DPPO-ID': 'dppo_id',
}

def load_metric(script_dir, transfer_metric, method, tag=''):
    episode_len_list = [9, 10, 11, 12, 13, 14, 15]
    ctrl = tag_ctrl_list[method]
    res = np.load(
        f'{script_dir}/../data/{ctrl}{tag}_gen_results.npy', allow_pickle=True).item()
    transfer_metric[method] = {'rmse': [], 'rmse_std': [], 'inference_time': []}
    for T in episode_len_list:
        T = '_'+str(T)
        transfer_metric[method]['rmse'].append(res[T]['mean_rmse'])
        transfer_metric[method]['rmse_std'].append(res[T]['std_rmse'])
    transfer_metric[method]['rmse'] = np.array(transfer_metric[method]['rmse'])
    transfer_metric[method]['rmse_std'] = np.array(transfer_metric[method]['rmse_std'])
    transfer_metric[method]['inference_time'] = np.mean(res['inference_time'])
    return transfer_metric

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
    data_folder_path = Path(notebook_dir) / controller_name / data_folder
    # print('data_folder_path', data_folder_path)
    assert data_folder_path.exists(), 'data_folder_path does not exist'

    # find all the subfolders in the data_folder_path
    subfolders = [f for f in data_folder_path.iterdir() if f.is_dir()]
    # print('subfolders', subfolders)
    # load the row 'rmse in the metrics.txt
    metrics = []
    traj_resutls = []
    timing_data = []
    for subfolder in subfolders:
        file_path = subfolder / 'metrics.txt'
        with file_path.open('r') as file:
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
        for file in subfolder.iterdir():
            if file.suffix == '.pkl':
                results = np.load(file, allow_pickle=True)
                traj_data = results['trajs_data']['obs'][0]
                traj_resutls.append(traj_data)

    traj_file_name = Path(f'traj_results_{controller_name}{additional}.npy')
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
