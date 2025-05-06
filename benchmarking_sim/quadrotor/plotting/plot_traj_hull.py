import os
import sys

import munch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

#from benchmarking_sim.quadrotor.plotting.plot_hull import lmpc_data_path, mpc_data_path
#from benchmarking_sim.quadrotor.plotting.plot_hull import ppo_data, ppo_data_path, sac_data_path, dppo_traj_data, \
#    dppo_data_path
from safe_control_gym.utils.configuration import ConfigFactory
from functools import partial
from safe_control_gym.utils.registration import make

# # get the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
script_path = os.path.dirname(os.path.realpath(__file__))

def plot_xz_trajectory_with_hull(ax, traj_data, label=None,
                                 traj_color='skyblue', hull_color='lightblue',
                                 linewidth=1.0, linestyle='-', alpha=0.5, padding_factor=1.1):
    '''Plot trajectories with convex hull showing variance over seeds.
    
    Args:
        ax (Axes): Matplotlib axes.
        traj_data (np.ndarray): Trajectory data of shape (num_seeds, num_steps, 6).
        padding_factor (float): Padding factor for the convex hull.
    '''
    num_seeds, num_steps, _ = traj_data.shape

    print('traj data shape:', traj_data.shape)
    mean_traj = np.mean(traj_data, axis=0)

    ax.plot(mean_traj[:, 0], mean_traj[:, 2], color=traj_color, linewidth=linewidth, linestyle=linestyle, label=label)
    # plot the hull
    for i in range(num_steps - 1):
        # plot the hull at a single step
        points_at_step = traj_data[:, i, [0, 2]]
        hull = ConvexHull(points_at_step)
        cent = np.mean(points_at_step, axis=0)  # center
        pts = points_at_step[hull.vertices]  # vertices
        poly = Polygon(padding_factor * (pts - cent) + cent,
                       closed=True,
                       capstyle='round',
                       facecolor=hull_color,
                       alpha=alpha)
        ax.add_patch(poly)

        # connecting consecutive convex hulls
        points_at_next_step = traj_data[:, i + 1, [0, 2]]
        points_connecting = np.concatenate([points_at_step, points_at_next_step], axis=0)
        hull_connecting = ConvexHull(points_connecting)
        cent_connecting = np.mean(points_connecting, axis=0)
        pts_connecting = points_connecting[hull_connecting.vertices]
        poly_connecting = Polygon(padding_factor * (pts_connecting - cent_connecting) + cent_connecting,
                                  closed=True,
                                  capstyle='round',
                                  facecolor=hull_color,
                                  alpha=alpha)
        ax.add_patch(poly_connecting)


#############################################
if len(sys.argv) > 1:
    if sys.argv[1] == 'rl':
        plot_name = 'RL'
    elif sys.argv[1] == 'mb':
        plot_name = 'Control-oriented'
if len(sys.argv) > 2:
    additional = sys.argv[2]
    # generalization = True if sys.argv[2] == 'gen' else False
# else:
#     generalization = False

# generalization = False
# generalization = True
generalization = False if additional != '11' else True
# plot_name = 'RL'
# plot_name = 'Control-oriented'
#############################################
# additional = '11'
# additional = '9'
# additional = '15'

# get the config
ALGO = 'mpc_acados'
SYS = 'quadrotor_2D_attitude'
# SYS = 'quadrotor_3D_attitude'
TASK = 'tracking'
# PRIOR = '200_hpo'
PRIOR = '100'
agent = 'quadrotor' if SYS in ['quadrotor_2D', 'quadrotor_2D_attitude', 'quadrotor_3D_attitude'] else SYS
SAFETY_FILTER = None

# check if the config file exists
assert os.path.exists(f'../config_overrides/{SYS}_{TASK}_{additional}.yaml'), \
    f'../config_overrides/{SYS}_{TASK}_{additional}.yaml does not exist'
assert os.path.exists(f'../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), \
    f'../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
if SAFETY_FILTER is None:
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', agent,
                    '--overrides',
                    f'../config_overrides/{SYS}_{TASK}_{additional}.yaml',
                    f'../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                    '--seed', '2',
                    '--use_gpu', 'True',
                    '--output_dir', f'./{ALGO}/results',
                    ]
fac = ConfigFactory()
fac.add_argument('--func', type=str, default='train', help='main function to run.')
fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
config = fac.merge()
# if generalization:
#     config.task_config.task_info.ilqr_traj_data = '/home/mingxuan/Repositories/scg_tsung/examples/lqr/ilqr_ref_traj_gen.npy'

# Create an environment
env_func = partial(make,
                   config.task,
                   seed=config.seed,
                   **config.task_config
                   )
random_env = env_func(gui=False)
X_GOAL = random_env.X_GOAL
# rmse_state_idx = [0, 2] if SYS == 'quadrotor_2D_attitude' else [0, 1, 2]
# print('X_GOAL.shape', X_GOAL.shape)
# print('X_GOAL', X_GOAL)
# exit()

# load Control-oriented data
pid_data_path = os.path.join(script_path, f'../data/traj_results_pid_{SYS}_{additional}.npy')
pid_traj_data = np.load(pid_data_path, allow_pickle=True)
print(pid_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

lqr_data_path = os.path.join(script_path, f'../data/traj_results_lqr_{SYS}_{additional}.npy')
lqr_traj_data = np.load(lqr_data_path, allow_pickle=True)
print(lqr_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

ilqr_data_path = os.path.join(script_path, f'../data/traj_results_ilqr_{SYS}_{additional}.npy')
ilqr_traj_data = np.load(ilqr_data_path, allow_pickle=True)
print(ilqr_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

lmpc_data_path = os.path.join(script_path, f'../data/traj_results_linear_mpc_acados_{SYS}_{additional}.npy')
lmpc_traj_data = np.load(lmpc_data_path, allow_pickle=True)
print(lmpc_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

mpc_data_path = os.path.join(script_path, f'../data/traj_results_mpc_acados_{SYS}_{additional}.npy')
mpc_traj_data = np.load(mpc_data_path, allow_pickle=True)
print(mpc_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

fmpc_data_path = os.path.join(script_path, f'../data/traj_results_fmpc_{SYS}_{additional}.npy')
fmpc_traj_data = np.load(fmpc_data_path, allow_pickle=True)
print(fmpc_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

gpmpc_data_path = os.path.join(script_path, f'../data/traj_results_gpmpc_acados_TP_{SYS}_{additional}.npy')
gpmpc_traj_data = np.load(gpmpc_data_path, allow_pickle=True)
print(gpmpc_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

ppo_data_path = os.path.join(script_path, f'../data/traj_results_ppo_{additional}.npy')
# print(f"File exists: {os.path.exists(ppo_data_path)}")
ppo_data = np.load(ppo_data_path, allow_pickle=True).item()
ppo_traj_data = np.array(ppo_data['obs'])
print(ppo_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

sac_data_path = os.path.join(script_path, f'../data/traj_results_sac_{additional}.npy')
sac_data = np.load(sac_data_path, allow_pickle=True).item()
sac_traj_data = np.array(sac_data['obs'])
print(sac_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

dppo_data_path = os.path.join(script_path, f'../data/traj_results_dppo_{additional}.npy')
dppo_data = np.load(dppo_data_path, allow_pickle=True).item()
dppo_traj_data = np.array(dppo_data['obs'])
print(dppo_traj_data.shape)  # (10, 541, 6) seed, time_step, obs

def compute_rmse(traj, ref, idx='xz'):
    if idx == 'xyz':
        state_idx = [0, 2, 4]
    elif idx == 'xy':
        state_idx = [0, 2]
    elif idx == 'xz':
        state_idx = [0, 2]
    min_length = min(traj.shape[0], ref.shape[0])
    traj = traj[:min_length]
    ref = ref[:min_length]
    error = np.sum((traj[:, state_idx] - ref[:, state_idx]) ** 2, axis=1)
    instant_error = np.sqrt(error)
    rmse = np.sqrt(np.mean(error, axis=0))
    return rmse, instant_error

def compute_mean_rmse(traj_data, ref, ctrl=None, SYS='quadrotor_2D_attitude'):
    # result_dict = {}
    if SYS == 'quadrotor_2D_attitude':
        rmse = [compute_rmse(traj_data[i], ref)[0] for i in range(traj_data.shape[0])]
        mean_rmse = np.mean(rmse)
        std_rmse = np.std(rmse)
        mean_error = np.mean(np.array([compute_rmse(traj_data[i], ref)[1] for i in range(traj_data.shape[0])]), axis=0)
        std_error = np.std(np.array([compute_rmse(traj_data[i], ref)[1] for i in range(traj_data.shape[0])]), axis=0)
        if ctrl is not None:
            print(f'RMSE {ctrl}: {mean_rmse} +/- {std_rmse}')
        return mean_rmse, std_rmse, mean_error, std_error
    elif SYS == 'quadrotor_3D_attitude':
        rmse_full = [compute_rmse(traj_data[i], ref, 'xyz')[0] for i in range(traj_data.shape[0])]
        mean_rmse_full, std_rmse_full = np.mean(rmse_full), np.std(rmse_full)
        mean_error_full = np.mean(np.array([compute_rmse(traj_data[i], ref, 'xyz')[1] for i in range(traj_data.shape[0])]), axis=0)
        std_error_full = np.std(np.array([compute_rmse(traj_data[i], ref, 'xyz')[1] for i in range(traj_data.shape[0])]), axis=0)
        rmse_xy = [compute_rmse(traj_data[i], ref, 'xy')[0] for i in range(traj_data.shape[0])]
        mean_rmse_xy, std_rmse_xy = np.mean(rmse_xy), np.std(rmse_xy)
        mean_error_xy = np.mean(np.array([compute_rmse(traj_data[i], ref, 'xy')[1] for i in range(traj_data.shape[0])]), axis=0)
        std_error_xy = np.std(np.array([compute_rmse(traj_data[i], ref, 'xy')[1] for i in range(traj_data.shape[0])]), axis=0)
        rmse_xz = [compute_rmse(traj_data[i], ref, 'xz')[0] for i in range(traj_data.shape[0])]
        mean_rmse_xz, std_rmse_xz = np.mean(rmse_xz), np.std(rmse_xz)
        mean_error_xz = np.mean(np.array([compute_rmse(traj_data[i], ref, 'xz')[1] for i in range(traj_data.shape[0])]), axis=0)
        std_error_xz = np.std(np.array([compute_rmse(traj_data[i], ref, 'xz')[1] for i in range(traj_data.shape[0])]), axis=0)
        if ctrl is not None:
            print(f'RMSE {ctrl} full: {mean_rmse_full} +/- {std_rmse_full}')
            print(f'RMSE {ctrl} xy: {mean_rmse_xy} +/- {std_rmse_xy}')
            print(f'RMSE {ctrl} xz: {mean_rmse_xz} +/- {std_rmse_xz}')
        return mean_rmse_full, std_rmse_full, mean_error_full, std_error_full, \
               mean_rmse_xy, std_rmse_xy, mean_error_xy, std_error_xy, \
               mean_rmse_xz, std_rmse_xz, mean_error_xz, std_error_xz

if SYS == 'quadrotor_2D_attitude':
    mean_rmse_pid, std_rmse_pid, mean_error_pid, std_error_pid = compute_mean_rmse(pid_traj_data, X_GOAL, 'PID')
    mean_rmse_lqr, std_rmse_lqr, mean_error_lqr, std_error_lqr = compute_mean_rmse(lqr_traj_data, X_GOAL, 'LQR')
    mean_rmse_ilqr, std_rmse_ilqr, mean_error_ilqr, std_error_ilqr = compute_mean_rmse(ilqr_traj_data, X_GOAL, 'iLQR')
    mean_rmse_gpmpc, std_rmse_gpmpc, mean_error_gpmpc, std_error_gpmpc = compute_mean_rmse(gpmpc_traj_data, X_GOAL, 'GP-MPC')
    mean_rmse_lmpc, std_rmse_lmpc, mean_error_lmpc, std_error_lmpc = compute_mean_rmse(lmpc_traj_data, X_GOAL, 'Linear MPC')
    mean_rmse_mpc, std_rmse_mpc, mean_error_mpc, std_error_mpc = compute_mean_rmse(mpc_traj_data, X_GOAL, 'MPC')
    mean_rmse_fmpc, std_rmse_fmpc, mean_error_fmpc, std_error_fmpc = compute_mean_rmse(fmpc_traj_data, X_GOAL, 'F-MPC')
    mean_rmse_ppo, std_rmse_ppo, mean_error_ppo, std_error_ppo = compute_mean_rmse(ppo_traj_data, X_GOAL, 'PPO')
    mean_rmse_sac, std_rmse_sac, mean_error_sac, std_error_sac = compute_mean_rmse(sac_traj_data, X_GOAL, 'SAC')
    mean_rmse_dppo, std_rmse_dppo, mean_error_dppo, std_error_dppo = compute_mean_rmse(dppo_traj_data, X_GOAL, 'DPPO')
elif SYS == 'quadrotor_3D_attitude':
    mean_rmse_pid_full, std_rmse_pid_full, mean_error_pid_full, std_error_pid_full, \
    mean_rmse_pid_xy, std_rmse_pid_xy, mean_error_pid_xy, std_error_pid_xy, \
    mean_rmse_pid_xz, std_rmse_pid_xz, mean_error_pid_xz, std_error_pid_xz = compute_mean_rmse(pid_traj_data, X_GOAL, 'PID', SYS)
# exit()
##################################################
# # plotting trajectory
# gpmpc_color = 'blue'
# # gpmpc_hull_color = 'lightskyblue'
# gpmpc_hull_color = 'cornflowerblue'
pid_color = 'gray'
pid_hull_color = 'lightgray'
ilqr_color = 'slateblue'
ilqr_hull_color = 'slateblue'
lqr_color = 'blueviolet'
lqr_hull_color = 'blueviolet'

# dppo_color = 'cyan'
# dppo_hull_color = 'lightcyan'
# ppo_color = 'orange'
# ppo_hull_color = 'peachpuff'
# sac_color = 'green'
# sac_hull_color = 'lightgreen'
ref_color = 'black'
# linear_mpc_color = 'purple'
# linear_mpc_hull_color = 'violet'
# mpc_color = 'red'
# mpc_hull_color = 'salmon'
gpmpc_color = 'royalblue'
gpmpc_hull_color = 'cornflowerblue'
lmpc_color = 'green'
lmpc_hull_color = 'lightgreen'
mpc_color = 'cadetblue'
mpc_hull_color = 'cadetblue'
fmpc_color = 'darkblue'
fmpc_hull_color = 'darkblue'

ppo_color = 'darkorange'
ppo_hull_color = 'moccasin'
sac_color = 'red'
sac_hull_color = 'salmon'
dppo_color = 'pink'
dppo_hull_color = 'lavenderblush'


plot_colors = {
    'GP-MPC': 'royalblue',
    'PPO': 'darkorange',
    'SAC': 'red',
    'DPPO': 'pink',
    'PID': 'darkgray',
    'Linear MPC': 'green',
    'Nonlinear MPC': 'cadetblue',
    'iLQR': 'slateblue',
    'LQR': 'blueviolet',
    'F-MPC': 'darkblue',
    'MAX': 'none',
    'MIN': 'none',
}

##################################################
# plot tracking error plot


plot_std_tracking_error = True
# plot_std_tracking_error = False
s = 2
fig, ax = plt.subplots(figsize=(5, 3))
# adjust the distance between title and the plot
# fig.subplots_adjust(top=0.2)
time_axis = np.arange(0, mean_error_pid.shape[0])
dt = 1/60
time_axis = time_axis * dt
if plot_name == 'RL':
    ax.plot(time_axis, mean_error_ppo, color=plot_colors['PPO'], label='PPO')
    ax.plot(time_axis, mean_error_sac, color=plot_colors['SAC'], label='SAC')
    ax.plot(time_axis, mean_error_dppo, color=plot_colors['DPPO'], label='DPPO')
    if plot_std_tracking_error:
        ax.fill_between(time_axis, mean_error_ppo - s * std_error_ppo, mean_error_ppo + s * std_error_ppo, color=plot_colors['PPO'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_sac - s * std_error_sac, mean_error_sac + s * std_error_sac, color=plot_colors['SAC'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_dppo - s * std_error_dppo, mean_error_dppo + s * std_error_dppo, color=plot_colors['DPPO'], alpha=0.2)
    ax.legend(ncol=1)
elif plot_name == 'Control-oriented':
    ax.plot(time_axis, mean_error_pid, color=plot_colors['PID'], label='PID')
    ax.plot(time_axis, mean_error_lqr, color=plot_colors['LQR'], label='LQR')
    ax.plot(time_axis, mean_error_ilqr, color=plot_colors['iLQR'], label='iLQR')
    ax.plot(time_axis, mean_error_lmpc, color=plot_colors['Linear MPC'], label='Linear MPC')
    ax.plot(time_axis, mean_error_mpc, color=plot_colors['Nonlinear MPC'], label='Nonlinear MPC')
    ax.plot(time_axis, mean_error_fmpc, color=plot_colors['F-MPC'], label='F-MPC')
    ax.plot(time_axis, mean_error_gpmpc, color=plot_colors['GP-MPC'], label='GP-MPC')
    if plot_std_tracking_error:
        ax.fill_between(time_axis, mean_error_pid - s * std_error_pid, mean_error_pid + s * std_error_pid, color=plot_colors['PID'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_lqr - s * std_error_lqr, mean_error_lqr + s * std_error_lqr, color=plot_colors['LQR'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_ilqr - s * std_error_ilqr, mean_error_ilqr + s * std_error_ilqr, color=plot_colors['iLQR'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_lmpc - s * std_error_lmpc, mean_error_lmpc + s * std_error_lmpc, color=plot_colors['Linear MPC'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_mpc - s * std_error_mpc, mean_error_mpc + s * std_error_mpc, color=plot_colors['Nonlinear MPC'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_gpmpc - s * std_error_gpmpc, mean_error_gpmpc + s * std_error_gpmpc, color=plot_colors['GP-MPC'], alpha=0.2)
    ax.legend(ncol=2)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Tracking error [m]')
ax.set_ylim(-0.05, 0.4)
# title_fontsize = 20
if additional == '11':
    fig.suptitle(f'Tracking error ({plot_name})',)
else:
    if additional == '9':
        fig.suptitle(f'Tracking error (faster) ({plot_name})',)
    elif additional == '15':
        fig.suptitle(f'Tracking error (slower) ({plot_name})',)

fig.tight_layout()
# plt.show()
file_name = f'tracking_error_{plot_name}_{additional}'
if not generalization:
    fig.savefig(os.path.join(script_path, file_name)+".pdf", bbox_inches='tight')
    print(f'Saved at {os.path.join(script_path, file_name)}.pdf')
    fig.savefig(os.path.join(script_path, f'{file_name}'+".png"), bbox_inches='tight')
    print(f'Saved at {os.path.join(script_path, f"{file_name}"+".png")}')
else:
    fig.savefig(os.path.join(script_path, f'{file_name}'+".pdf"), bbox_inches='tight')
    print(f'Saved at {os.path.join(script_path, f"{file_name}"+".pdf")}')
    fig.savefig(os.path.join(script_path, f'{file_name}'+".png"), bbox_inches='tight')
    print(f'Saved at {os.path.join(script_path, f"{file_name}"+".png")}')

##################################################
# plot the state path x, z [0, 2]
title_fontsize = 20
legend_fontsize = 12
axis_label_fontsize = 12
axis_tick_fontsize = 12
dummy = int(X_GOAL.shape[0] / 2)
fig, ax = plt.subplots(figsize=(8, 4))
# adjust the distance between title and the plot
fig.subplots_adjust(top=0.2)

# plot the convex hull of each steps
k = 1.1  # padding factor
alpha = 0.02

if plot_name == 'RL':
    plot_xz_trajectory_with_hull(ax, sac_traj_data, label='SAC',
                                 traj_color=plot_colors['SAC'], hull_color=plot_colors['SAC'],
                                 linewidth=2.0, alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, ppo_traj_data, label='PPO',
                                 traj_color=plot_colors['PPO'], hull_color=plot_colors['PPO'],
                                 linewidth=2.0, alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, dppo_traj_data, label='DPPO',
                                 traj_color=plot_colors['DPPO'], hull_color=plot_colors['DPPO'],
                                 linewidth=2.0, alpha=alpha, padding_factor=k)
elif plot_name == 'Control-oriented':
    plot_xz_trajectory_with_hull(ax, pid_traj_data, label='PID',
                                    traj_color=plot_colors['PID'], hull_color=plot_colors['PID'],
                                    linewidth=2.0, alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, lqr_traj_data, label='LQR',
                                    traj_color=lqr_color, hull_color=lqr_hull_color, 
                                    linewidth=2.0, alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, ilqr_traj_data, label='iLQR',
                                    traj_color=ilqr_color, hull_color=ilqr_hull_color, 
                                    linewidth=2.0, alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, gpmpc_traj_data, label='GP-MPC',
                                 traj_color=plot_colors['GP-MPC'], hull_color=plot_colors['GP-MPC'],
                                 linewidth=2.0, alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, lmpc_traj_data, label='Linear MPC',
                                 traj_color=plot_colors['Linear MPC'], hull_color=plot_colors['Linear MPC'],
                                 linewidth=2.0, alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, mpc_traj_data, label='Nonlinear MPC',
                                 traj_color=plot_colors['Nonlinear MPC'], hull_color=plot_colors['Nonlinear MPC'],
                                 linewidth=2.0, alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, fmpc_traj_data, label='F-MPC',
                                    traj_color=plot_colors['F-MPC'], hull_color=plot_colors['F-MPC'],
                                    linewidth=2.0, alpha=alpha, padding_factor=k)   

ax.plot(X_GOAL[:dummy, 0], X_GOAL[:dummy, 2], color=ref_color, linestyle='-.', linewidth=1., label='Reference')
# ax.plot()
ax.set_xlabel('$x$ [m]', fontsize=axis_label_fontsize)
ax.set_ylabel('$z$ [m]', fontsize=axis_label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=axis_tick_fontsize)
# ax.set_title('State path in $x$-$z$ plane')
# set the super title
if additional == '11':
    fig.suptitle(f'Evaluation ({plot_name})', fontsize=title_fontsize)
else:
    if additional == '9':
        fig.suptitle(f'Generalization (faster) ({plot_name} )', fontsize=title_fontsize)
    elif additional == '15':
        fig.suptitle(f'Generalization (slower) ({plot_name} )', fontsize=title_fontsize)
# fig.suptitle(f'Evaluation ({plot_name})', fontsize=title_fontsize)
# fig.suptitle(f'Generalization (slower) ({plot_name})', fontsize=title_fontsize)
ax.set_ylim(0.35, 1.85)
ax.set_xlim(-1.6, 1.6)
fig.tight_layout()

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
# order = [3, 1, 2, 0]
# order = [0, 4, 6, 1, 5, 2, 3]
order = np.arange(len(labels))

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=3, loc='upper center', fontsize=legend_fontsize)
#ax.legend(ncol=5, loc='upper center', fontsize=legend_fontsize)

'''
NOTE: The current color choice is not ideal in the sense that 
overlapping the same color will make the color darker.
Therefore, alpha of each convex hull is set to 1.0. This will 
resutls in different convex hulls overlapping each other and 
the one in the bottom will not be visible.
'''
# plt.show()

# if not generalization:
if additional == '11':
    fig.savefig(os.path.join(script_path, f'{plot_name}_xz_path_performance_{additional}.pdf'), bbox_inches='tight')
    print(f'Saved at {os.path.join(script_path, f"{plot_name}_xz_path_performance_{additional}.pdf")}')
    fig.savefig(os.path.join(script_path, f'{plot_name}_xz_path_performance_{additional}.png'), bbox_inches='tight') 
    print(f'Saved at {os.path.join(script_path, f"{plot_name}_xz_path_performance_{additional}.png")}')
else:
    fig.savefig(os.path.join(script_path, f'{plot_name}_xz_path_generalization_{additional}.pdf'), bbox_inches='tight')
    print(f'Saved at {os.path.join(script_path, f"{plot_name}_xz_path_generalization_{additional}.pdf")}')
    fig.savefig(os.path.join(script_path, f'{plot_name}_xz_path_generalization_{additional}.png'), bbox_inches='tight')
    print(f'Saved at {os.path.join(script_path, f"{plot_name}_xz_path_generalization_{additional}.png")}')
