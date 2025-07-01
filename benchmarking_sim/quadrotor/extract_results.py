import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from benchmarking_sim.quadrotor.benchmark_util.utils import plot_colors

notebook_dir = os.path.dirname(os.path.abspath('__file__'))
print('notebook_dir', notebook_dir)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
s = 2 # times of std


from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
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


def plot_trajectory(notebook_dir, data_folder, title, ctrl, 
                    SYS='quadrotor_2D_attitude', 
                    additional=''):
    from safe_control_gym.utils.configuration import ConfigFactory
    from functools import partial
    from safe_control_gym.utils.registration import make
    #########################################################################
    # launch SCG to get reference trajectory X_GOAL
    ALGO = ctrl
    # SYS = 'quadrotor_2D_attitude'
    TASK = 'tracking'
    # PRIOR = '200_hpo'
    PRIOR = '100'
    agent = 'quadrotor' if SYS in ['quadrotor_2D', 'quadrotor_2D_attitude', 'quadrotor_3D_attitude'] else SYS
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
    controller_name = ctrl
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
    ref_color = plot_colors['Reference']
    fmpc_color = plot_colors['F-MPC']

    # Adjust hull colors using transparency
    hull_alpha = 0.3

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
    try:
        plot_xz_trajectory_with_hull(ax, fmpc_traj_data, label='FMPC',
                                     traj_color=fmpc_color, hull_color=fmpc_color,
                                     alpha=hull_alpha, padding_factor=k)
    except Exception as e:
        print(f'Error plotting trajectory with hull: {e}')
        for i in range(fmpc_traj_data.shape[0]):
            ax.plot(fmpc_traj_data[i, :, 0], fmpc_traj_data[i, :, 2], 
                    color=fmpc_color, alpha=alpha, linewidth=0.5)

    ax.legend(ncol=5, loc='upper center', fontsize=legend_fontsize)

    fig.savefig(os.path.join(fmpc_data_path, f'xz_path_performance{SYS}{additional}.png'), dpi=300, bbox_inches='tight')
    print(f'saved to {fmpc_data_path}/xz_path_performance{SYS}{additional}.png')
    # save data
    np.save(os.path.join(fmpc_data_path, f'traj_results_{ctrl}_{SYS}{additional}.npy'), fmpc_traj_data)
    print(f'traj data saved to {fmpc_data_path}/traj_results_{ctrl}_{SYS}{additional}.npy')

    # copy the data file to the results folder
    results_folder = os.path.join(notebook_dir, 'data')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    os.system(f'cp {fmpc_data_path}/traj_results_{ctrl}_{SYS}{additional}.npy {results_folder}')
    print(f'copied to {results_folder}/traj_results_{ctrl}_{SYS}{additional}.npy')
    
def extract_rollouts(notebook_dir, data_folder, controller_name, additional=''):
    # print('notebook_dir', notebook_dir)
    data_folder_path = os.path.join(notebook_dir, controller_name, data_folder)
    # print('data_folder_path', data_folder_path)
    assert os.path.exists(data_folder_path), f'data_folder_path {data_folder_path} does not exist'

    # find all the subfolders in the data_folder_path
    subfolders = [f.path for f in os.scandir(data_folder_path) if f.is_dir()]
    # sort the subfolders
    subfolders.sort()
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

    # print('traj_results.shape', traj_resutls)
    # if type(traj_resutls)
    rmse_mean_mpc = np.mean(metrics)
    rmse_std_mpc = np.std(metrics)
    print(f'rmse_{controller_name}{additional}', rmse_mean_mpc, rmse_std_mpc)
    return metrics, timing_data

    traj_resutls = np.array(traj_resutls)
    traj_file_name = f'traj_results_{controller_name}{additional}.npy'
    np.save(traj_file_name, traj_resutls)
    print('traj_results.shape', traj_resutls.shape)
    # print('metrics', metrics)
    return traj_resutls, metrics

if len(sys.argv) > 1:
    ctrl = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ''
else:
    # ctrl = 'pid'
    # ctrl = 'pid'
    # ctrl = 'ilqr'
    # ctrl = 'fmpc'
    # ctrl = 'linear_mpc'
    # ctrl = 'linear_mpc_acados'
    # ctrl = 'mpc_acados'
    ctrl = 'gpmpc_acados_TP'
    tag = '_handtune'
gp_model_tag = f'_{tag}'
# gp_model_tag = tag
SYS = 'quadrotor_2D_attitude'
# SYS = 'quadrotor_3D_attitude'

# for additional in ['_11', '_12', '_13', '_14', '_15']:
results = {}
for additional in ['9', '10', '11', '12', '13', '14', '15']:
# for additional in ['9',]:
    additional = '_' + additional
    data_folder = f'results_rollout_{SYS}{additional}/temp'
    if ctrl in ['gpmpc_acados_TP']:
        GPMPC_option = f'{gp_model_tag}'
        data_folder = f'results/{GPMPC_option}_rollout_{SYS}{additional}/temp'
    # traj_resutls, metrics = extract_rollouts(notebook_dir, data_folder, ctrl, additional)
    if additional == '_11':
        metrics, timing_data = extract_rollouts(notebook_dir, data_folder, ctrl, additional)
    else:
        metrics, _ = extract_rollouts(notebook_dir, data_folder, ctrl, additional)
    mean_rmse = np.mean(metrics)
    std_rmse = np.std(metrics)
    results[additional] = {'mean_rmse': mean_rmse, 'std_rmse': std_rmse}
results['inference_time'] = np.mean(timing_data)
print('mean inference time:', results['inference_time'])
print('results', results)
np.save(f'data/{ctrl}{tag}_gen_results.npy', results)
# print('metrics', metrics)
# time_vector = (np.squeeze(timing_data)).flatten()
# mean_exec_time = np.mean(time_vector)
# std_exec_time = np.std(time_vector)
# max_exec_time = np.max(time_vector)
# print('Mean execution time:', mean_exec_time)
# print('Max execution time:', max_exec_time)
# print('Std execution time:', std_exec_time)
# sp_plot_inf_time = mean_exec_time # save for later, spider plot

additional = '_11'
data_folder = f'results_rollout_{SYS}{additional}/temp'
if ctrl in ['gpmpc_acados_TP']:
        GPMPC_option = gp_model_tag
        data_folder = f'results/{GPMPC_option}_rollout_{SYS}{additional}/temp'
plot_trajectory(notebook_dir, data_folder, 'Evaluation', ctrl, SYS, additional)

additional = '_15'
data_folder = f'results_rollout_{SYS}{additional}/temp'
if ctrl in ['gpmpc_acados_TP']:
        GPMPC_option = gp_model_tag
        data_folder = f'results/{GPMPC_option}_rollout_{SYS}{additional}/temp'
plot_trajectory(notebook_dir, data_folder, 'Generalization (slower)', ctrl, SYS, additional)

additional = '_9'
data_folder = f'results_rollout_{SYS}{additional}/temp'
if ctrl in ['gpmpc_acados_TP']:
        GPMPC_option = gp_model_tag
        data_folder = f'results/{GPMPC_option}_rollout_{SYS}{additional}/temp'
plot_trajectory(notebook_dir, data_folder, 'Generalization (faster)', ctrl, SYS, additional)
# additional = '_safety'
# ctrl = 'mpc_acados'
# data_folder = f'results_rollout_{SYS}{additional}/temp'
# plot_trajectory(notebook_dir, data_folder, 'Safety', ctrl, SYS, additional)

# additional = '_safety_noiseless'
# ctrl = 'mpc_acados'
# data_folder = f'results_rollout_{SYS}{additional}/temp'
# plot_trajectory(notebook_dir, data_folder, 'Safety', ctrl, SYS, additional)

# additional = '_safety_noiseless'
# ctrl = 'gpmpc_acados_TP'
# GPMPC_option = f'_handtune_safety_noiseless'
# # results/_handtune_safety_noiseless_rollout_quadrotor_2D_attitude_safety_noiseless
# data_folder = f'results/{GPMPC_option}_rollout_{SYS}{additional}/temp'
# # data_folder = f'results/_handtune_safety_noiseless_rollout_quadrotor_2D_attitude_safety_noiseless'
# plot_trajectory(notebook_dir, data_folder, 'Safety', ctrl, SYS, additional)

# additional = '_safety'
# ctrl = 'gpmpc_acados_TP'
# GPMPC_option = f'_handtune_safety'
# # results/_handtune_safety_noiseless_rollout_quadrotor_2D_attitude_safety_noiseless
# data_folder = f'results/{GPMPC_option}_rollout_{SYS}{additional}/temp'
# # data_folder = f'results/_handtune_safety_noiseless_rollout_quadrotor_2D_attitude_safety_noiseless'
# plot_trajectory(notebook_dir, data_folder, 'Safety', ctrl, SYS, additional)
