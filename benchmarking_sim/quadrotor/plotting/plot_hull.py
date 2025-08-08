
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

from safe_control_gym.utils.configuration import ConfigFactory
from functools import partial
from safe_control_gym.utils.registration import make

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


#############################################
if len(sys.argv) > 1:
    if sys.argv[1] == 'rl':
        plot_name = 'RL'
    elif sys.argv[1] == 'mb':
        plot_name = 'Model-based'
if len(sys.argv) > 2:
    generalization = True if sys.argv[2] == 'gen' else False
else:
    generalization = False


# generalization = False
# generalization = True
# plot_name = 'RL'
# plot_name = 'Model-based'
#############################################



# get the config
ALGO = 'mpc_acados'
SYS = 'quadrotor_2D_attitude'
TASK = 'tracking'
# PRIOR = '200_hpo'
PRIOR = '100'
agent = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' else SYS
SAFETY_FILTER = None

# check if the config file exists
assert os.path.exists(f'../config_overrides/{SYS}_{TASK}.yaml'), f'../config_overrides/{SYS}_{TASK}.yaml does not exist'
assert os.path.exists(f'../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), f'../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
if SAFETY_FILTER is None:
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', agent,
                    '--overrides',
                        f'../config_overrides/{SYS}_{TASK}.yaml',
                        f'../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                    '--seed', '2',
                    '--use_gpu', 'True',
                    '--output_dir', f'./{ALGO}/results',
                        ]
fac = ConfigFactory()
fac.add_argument('--func', type=str, default='train', help='main function to run.')
fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
# merge config and create output directory
config = fac.merge()
if generalization:
    config.task_config.task_info.ilqr_traj_data = '/home/mingxuan/Repositories/scg_tsung/examples/lqr/ilqr_ref_traj_gen.npy'
# Create an environment
env_func = partial(make,
                    config.task,
                    seed=config.seed,
                    **config.task_config
                    )
random_env = env_func(gui=False)
X_GOAL = random_env.X_GOAL
# print('X_GOAL.shape', X_GOAL.shape)

# get the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

script_path = os.path.dirname(os.path.realpath(__file__))


if not generalization:
    gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados/results/200_300_aggresive'
if generalization:
    gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados/results/100_400_rollout/temp'
# gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados/results/200_300_rti/temp'
# get all directories in the gp_model_path
gp_model_dirs = [d for d in os.listdir(gp_model_path) if os.path.isdir(os.path.join(gp_model_path, d))]
gp_model_dirs = [os.path.join(gp_model_path, d) for d in gp_model_dirs]

traj_data_name = 'gpmpc_acados_data_quadrotor_traj_tracking.pkl'
data_name = [os.path.join(d, traj_data_name) for d in gp_model_dirs]

# print(data_name)
# data = np.load(data_name[0], allow_pickle=True)
# print(data.keys())
# print(data['trajs_data'].keys())
# print(data['trajs_data']['obs'][0].shape) # (541, 6)
data = []
for d in data_name:
    data.append(np.load(d, allow_pickle=True))
gpmpc_traj_data = [d['trajs_data']['obs'][0] for d in data]
gpmpc_traj_data = np.array(gpmpc_traj_data)
if generalization:
    np.save('gpmpc_traj_data_gen.npy', gpmpc_traj_data)
else:
    np.save('gpmpc_traj_data.npy', gpmpc_traj_data)
print(gpmpc_traj_data.shape) # (10, 541, 6) seed, time_step, obs
# take average of all seeds
mean_traj_data = np.mean(gpmpc_traj_data, axis=0)
print(mean_traj_data.shape) # (mean_541, 6)

### plot the ilqr data
if not generalization:
    ilqr_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/ilqr/results/temp'
if generalization:
    ilqr_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/ilqr/results_rollout/temp'

ilqr_data_dirs = [d for d in os.listdir(ilqr_data_path) if os.path.isdir(os.path.join(ilqr_data_path, d))]
ilqr_traj_data_name = 'ilqr_data_quadrotor_traj_tracking.pkl'
ilqr_traj_data_name = [os.path.join(d, ilqr_traj_data_name) for d in ilqr_data_dirs]

ilqr_data = []
for d in ilqr_traj_data_name:
    ilqr_data.append(np.load(os.path.join(ilqr_data_path, d), allow_pickle=True))
ilqr_traj_data = [d['trajs_data']['obs'][0] for d in ilqr_data]
ilqr_traj_data = np.array(ilqr_traj_data)
print(ilqr_traj_data.shape) # (10, 541, 6) seed, time_step, obs
# take average of all seeds
ilqr_mean_traj_data = np.mean(ilqr_traj_data, axis=0)
print(ilqr_mean_traj_data.shape) # (mean_541, 6)

### plot the linear mpc data
if not generalization:
    lmpc_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/linear_mpc/results/temp'
if generalization:
    lmpc_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/linear_mpc/results_rollout/temp'
lmpc_data_dirs = [d for d in os.listdir(lmpc_data_path) if os.path.isdir(os.path.join(lmpc_data_path, d))]
lmpc_traj_data_name = 'linear_mpc_data_quadrotor_traj_tracking.pkl'
lmpc_traj_data_name = [os.path.join(d, lmpc_traj_data_name) for d in lmpc_data_dirs]

lmpc_data = []
for d in lmpc_traj_data_name:
    lmpc_data.append(np.load(os.path.join(lmpc_data_path, d), allow_pickle=True))
lmpc_traj_data = [d['trajs_data']['obs'][0] for d in lmpc_data]
lmpc_traj_data = np.array(lmpc_traj_data)
print(lmpc_traj_data.shape) # (10, 541, 6) seed, time_step, obs
# take average of all seeds
lmpc_mean_traj_data = np.mean(lmpc_traj_data, axis=0)
print(lmpc_mean_traj_data.shape) # (mean_541, 6)

### plot the mpc data
if not generalization:
    mpc_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/mpc_acados/results/temp'
if generalization:
    mpc_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/mpc_acados/results_rollout/temp'
mpc_data_dirs = [d for d in os.listdir(mpc_data_path) if os.path.isdir(os.path.join(mpc_data_path, d))]
mpc_traj_data_name = 'mpc_acados_data_quadrotor_traj_tracking.pkl'
mpc_traj_data_name = [os.path.join(d, mpc_traj_data_name) for d in mpc_data_dirs]

mpc_data = []
for d in mpc_traj_data_name:
    mpc_data.append(np.load(os.path.join(mpc_data_path, d), allow_pickle=True))
mpc_traj_data = [d['trajs_data']['obs'][0] for d in mpc_data]
mpc_traj_data = np.array(mpc_traj_data)
print(mpc_traj_data.shape) # (10, 541, 6) seed, time_step, obs
# take average of all seeds
mpc_mean_traj_data = np.mean(mpc_traj_data, axis=0)
print(mpc_mean_traj_data.shape) # (mean_541, 6)

# load ppo and sac data
if not generalization:
    ppo_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/traj_results_ppo.npy'
else:
    ppo_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/gen_traj_results_ppo.npy'
ppo_data = np.load(ppo_data_path, allow_pickle=True).item()
print(ppo_data.keys()) # (x, 541, 6) seed, time_step, obs
print(ppo_data['obs'][0].shape)
ppo_traj_data = np.array(ppo_data['obs'])
print(ppo_traj_data.shape) # (10, 541, 6) seed, time_step, obs


if not generalization:
    sac_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/traj_results_sac.npy'
else:
    sac_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/gen_traj_results_sac.npy'
sac_data = np.load(sac_data_path, allow_pickle=True).item()
print(sac_data.keys()) # (x, 541, 6) seed, time_step, obs
print(sac_data['obs'][0].shape)
sac_traj_data = np.array(sac_data['obs'])
print(sac_traj_data.shape) # (10, 541, 6) seed, time_step, obs


if not generalization:
    dppo_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/traj_results_dppo.npy'
else:
    dppo_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/gen_traj_results_dppo.npy'
dppo_data = np.load(dppo_data_path, allow_pickle=True).item()
print(dppo_data.keys()) # (x, 541, 6) seed, time_step, obs
print(dppo_data['obs'][0].shape)
dppo_traj_data = np.array(dppo_data['obs'])
print(dppo_traj_data.shape) # (10, 541, 6) seed, time_step, obs

##################################################
# # plotting trajectory
# gpmpc_color = 'blue'
# # gpmpc_hull_color = 'lightskyblue'
# gpmpc_hull_color = 'cornflowerblue'
ilqr_color = 'gray'
ilqr_hull_color = 'lightgray'
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
mpc_color = 'aqua'
mpc_hull_color = 'paleturquoise'

ppo_color = 'darkorange'
ppo_hull_color = 'moccasin'
sac_color = 'red'
sac_hull_color = 'salmon'
dppo_color = 'pink'
dppo_hull_color = 'lavenderblush'

##################################################
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
ax.set_ylim(0.35, 1.85)
ax.set_xlim(-1.6, 1.6)
fig.tight_layout()


# plot the convex hull of each steps
k = 1.1 # padding factor
alpha = 0.2

if plot_name == 'RL':
    plot_xz_trajectory_with_hull(ax, dppo_traj_data, label='DPPO',
                                    traj_color=dppo_color, hull_color=dppo_hull_color,
                                        alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, sac_traj_data, label='SAC', 
                                traj_color=sac_color, hull_color=sac_hull_color, 
                                alpha=alpha, padding_factor=k)
# plot_xz_trajectory_with_hull(ax, ilqr_traj_data, label='iLQR',
#                                 traj_color=ilqr_color, hull_color=ilqr_hull_color,
#                                     alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, ppo_traj_data, label='PPO',
                                    traj_color=ppo_color, hull_color=ppo_hull_color,
                                        alpha=alpha, padding_factor=k)
elif plot_name == 'Model-based':
    plot_xz_trajectory_with_hull(ax, lmpc_traj_data, label='Linear-MPC',
                                    traj_color=lmpc_color, hull_color=lmpc_hull_color,
                                    alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, mpc_traj_data, label='MPC',
                                 traj_color=mpc_color, hull_color=mpc_hull_color,
                                    alpha=alpha, padding_factor=k)
    plot_xz_trajectory_with_hull(ax, gpmpc_traj_data, label='GP-MPC',
                                  traj_color=gpmpc_color, hull_color=gpmpc_hull_color,
                                    alpha=alpha, padding_factor=k)
ax.legend(ncol=5, loc='upper center', fontsize=legend_fontsize)
'''
NOTE: The current color choice is not ideal in the sense that 
overlapping the same color will make the color darker.
Therefore, alpha of each convex hull is set to 1.0. This will 
resutls in different convex hulls overlapping each other and 
the one in the bottom will not be visible.
'''


if not generalization:
    fig.savefig(os.path.join(script_path, f'{plot_name}_xz_path_performance.png'), bbox_inches='tight')
    print(f'Saved at {os.path.join(script_path, f"{plot_name}_xz_path_performance.png")}')
else:
    fig.savefig(os.path.join(script_path, f'{plot_name}_xz_path_generalization.png'), bbox_inches='tight')
    print(f'Saved at {os.path.join(script_path, f"{plot_name}_xz_path_generalization.png")}')


