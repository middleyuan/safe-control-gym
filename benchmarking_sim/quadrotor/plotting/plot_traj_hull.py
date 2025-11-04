import pathlib
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from benchmarking_sim.quadrotor.benchmark_util.utils import (STEPS_PER_SECOND, plot_colors,
                                                             plot_xz_trajectory_with_hull)
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

sns.set_theme(style='whitegrid')
script_path = pathlib.Path(__file__).parent.resolve()
#############################################
if len(sys.argv) > 1:
    if sys.argv[1] == 'rl':
        plot_name = 'RL'
    elif sys.argv[1] == 'mb':
        plot_name = 'Control-oriented'
    if len(sys.argv) > 2:
        additional = sys.argv[2]
else:
    generalization = False
    additional = '11'  # default value for additional

generalization = False if additional != '11' else True
#############################################
ALGO = 'mpc_acados'
SYS = 'quadrotor_2D_attitude'
TASK = 'tracking'
PRIOR = '100'
agent = 'quadrotor' if SYS in ['quadrotor_2D', 'quadrotor_2D_attitude', 'quadrotor_3D_attitude'] else SYS
SAFETY_FILTER = None

# Check if the config file exists
assert (script_path / f'../config_overrides/{SYS}_{TASK}_{additional}.yaml').exists(), \
    f'{script_path / f".. / config_overrides / {SYS}_{TASK}_{additional}.yaml"} does not exist'
assert (script_path / f'../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml').exists(), \
    f'{script_path / f".. / config_overrides / {ALGO}_{SYS}_{TASK}_{PRIOR}.yaml"} does not exist'

if SAFETY_FILTER is None:
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', agent,
                    '--overrides',
                    str(script_path / f'../config_overrides/{SYS}_{TASK}_{additional}.yaml'),
                    str(script_path / f'../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'),
                    '--seed', '2',
                    '--use_gpu', 'True',
                    '--output_dir', str(script_path / f'./{ALGO}/results'),
                    ]
fac = ConfigFactory()
fac.add_argument('--func', type=str, default='train', help='main function to run.')
fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
config = fac.merge()

# Create an environment
env_func = partial(make,
                   config.task,
                   seed=config.seed,
                   **config.task_config
                   )
random_env = env_func(gui=False)
X_GOAL = random_env.X_GOAL
random_env.close()

# Load Control-oriented data
pid_data_path = script_path / f'../data/traj_results_pid_{SYS}_{additional}.npy'
pid_traj_data = np.load(pid_data_path, allow_pickle=True)
print(pid_traj_data.shape)

lqr_data_path = script_path / f'../data/traj_results_lqr_{SYS}_{additional}.npy'
lqr_traj_data = np.load(lqr_data_path, allow_pickle=True)
print(lqr_traj_data.shape)

ilqr_data_path = script_path / f'../data/traj_results_ilqr_{SYS}_{additional}.npy'
ilqr_traj_data = np.load(ilqr_data_path, allow_pickle=True)
print(ilqr_traj_data.shape)

lmpc_data_path = script_path / f'../data/traj_results_linear_mpc_acados_{SYS}_{additional}.npy'
lmpc_traj_data = np.load(lmpc_data_path, allow_pickle=True)
print(lmpc_traj_data.shape)

mpc_data_path = script_path / f'../data/traj_results_mpc_acados_{SYS}_{additional}.npy'
mpc_traj_data = np.load(mpc_data_path, allow_pickle=True)
print(mpc_traj_data.shape)

fmpc_data_path = script_path / f'../data/traj_results_fmpc_{SYS}_{additional}.npy'
fmpc_traj_data = np.load(fmpc_data_path, allow_pickle=True)
print(fmpc_traj_data.shape)

gpmpc_data_path = script_path / f'../data/traj_results_gpmpc_acados_TP_{SYS}_{additional}.npy'
gpmpc_traj_data = np.load(gpmpc_data_path, allow_pickle=True)
print(gpmpc_traj_data.shape)

ppo_data_path = script_path / f'../data/trajectory/nominal/traj_results_ppo_{additional}.npy'
ppo_data = np.load(ppo_data_path, allow_pickle=True).item()
ppo_traj_data = np.array(ppo_data['obs'])
print(ppo_traj_data.shape)

sac_data_path = script_path / f'../data/trajectory/nominal/traj_results_sac_{additional}.npy'
sac_data = np.load(sac_data_path, allow_pickle=True).item()
sac_traj_data = np.array(sac_data['obs'])
print(sac_traj_data.shape)

dppo_data_path = script_path / f'../data/trajectory/nominal/traj_results_dppo_{additional}.npy'
dppo_data = np.load(dppo_data_path, allow_pickle=True).item()
dppo_traj_data = np.array(dppo_data['obs'])
print(dppo_traj_data.shape)

ppo_mpc_data_path = script_path / f'../data/trajectory/nominal/traj_results_ppo_mpc_{additional}.npy'
ppo_mpc_data = np.load(ppo_mpc_data_path, allow_pickle=True).item()
ppo_mpc_traj_data = np.array(ppo_mpc_data['obs'])
print(ppo_mpc_traj_data.shape)


def compute_rmse_and_mean(traj_data, ref, ctrl=None):
    '''
    Compute RMSE and mean RMSE for trajectory data.

    Args:
        traj_data (np.ndarray): Trajectory data of shape (num_seeds, num_steps, state_dim).
        ref (np.ndarray): Reference trajectory of shape (num_steps, state_dim).
        ctrl (str): Controller name for printing RMSE.

    Returns:
        tuple: mean_rmse, std_rmse, mean_error, std_error
    '''
    state_idx = [0, 2]

    min_length = min(traj_data.shape[1], ref.shape[0])
    traj_data = traj_data[:, :min_length]
    ref = ref[:min_length]

    errors = np.sqrt(np.sum((traj_data[:, :, state_idx] - ref[:, state_idx]) ** 2, axis=2))
    rmse = np.sqrt(np.mean(errors ** 2, axis=1))
    mean_rmse = np.mean(rmse)
    std_rmse = np.std(rmse)
    mean_error = np.mean(errors, axis=0)
    std_error = np.std(errors, axis=0)

    if ctrl is not None:
        print(f'RMSE {ctrl}: {mean_rmse:.3f} +/- {std_rmse:.3f}')
    return mean_rmse, std_rmse, mean_error, std_error


# Update calls to the merged function
mean_rmse_pid, std_rmse_pid, mean_error_pid, std_error_pid = compute_rmse_and_mean(pid_traj_data, X_GOAL, 'Geometric Control')
mean_rmse_lqr, std_rmse_lqr, mean_error_lqr, std_error_lqr = compute_rmse_and_mean(lqr_traj_data, X_GOAL, 'LQR')
mean_rmse_ilqr, std_rmse_ilqr, mean_error_ilqr, std_error_ilqr = compute_rmse_and_mean(ilqr_traj_data, X_GOAL, 'iLQR')
mean_rmse_gpmpc, std_rmse_gpmpc, mean_error_gpmpc, std_error_gpmpc = compute_rmse_and_mean(gpmpc_traj_data, X_GOAL, 'GP-MPC')
mean_rmse_lmpc, std_rmse_lmpc, mean_error_lmpc, std_error_lmpc = compute_rmse_and_mean(lmpc_traj_data, X_GOAL, 'Linear MPC')
mean_rmse_mpc, std_rmse_mpc, mean_error_mpc, std_error_mpc = compute_rmse_and_mean(mpc_traj_data, X_GOAL, 'MPC')
mean_rmse_fmpc, std_rmse_fmpc, mean_error_fmpc, std_error_fmpc = compute_rmse_and_mean(fmpc_traj_data, X_GOAL, 'F-MPC')
mean_rmse_ppo, std_rmse_ppo, mean_error_ppo, std_error_ppo = compute_rmse_and_mean(ppo_traj_data, X_GOAL, 'PPO')
mean_rmse_sac, std_rmse_sac, mean_error_sac, std_error_sac = compute_rmse_and_mean(sac_traj_data, X_GOAL, 'SAC')
mean_rmse_dppo, std_rmse_dppo, mean_error_dppo, std_error_dppo = compute_rmse_and_mean(dppo_traj_data, X_GOAL, 'DPPO')
mean_rmse_ppo_mpc, std_rmse_ppo_mpc, mean_error_ppo_mpc, std_error_ppo_mpc = compute_rmse_and_mean(ppo_mpc_traj_data, X_GOAL, 'PPO-MPC')

# Adjust hull colors using transparency
hull_alpha = 0.3

# Plot tracking error plot
plot_std_tracking_error = True
# plot_std_tracking_error = False
s = 2
fig, ax = plt.subplots(figsize=(6, 4))
# adjust the distance between title and the plot
time_axis = np.arange(0, mean_error_pid.shape[0])
dt = 1 / 60
time_axis = time_axis * dt
if plot_name == 'RL':
    ax.plot(time_axis, mean_error_ppo, color=plot_colors['PPO'], label='PPO')
    ax.plot(time_axis, mean_error_sac, color=plot_colors['SAC'], label='SAC')
    ax.plot(time_axis, mean_error_dppo, color=plot_colors['DPPO'], label='DPPO')
    ax.plot(time_axis, mean_error_ppo_mpc, color=plot_colors['PPO-MPC'], label='PPO-MPC')
    if plot_std_tracking_error:
        ax.fill_between(time_axis, mean_error_ppo - s * std_error_ppo, mean_error_ppo + s * std_error_ppo, color=plot_colors['PPO'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_sac - s * std_error_sac, mean_error_sac + s * std_error_sac, color=plot_colors['SAC'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_dppo - s * std_error_dppo, mean_error_dppo + s * std_error_dppo, color=plot_colors['DPPO'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_ppo_mpc - s * std_error_ppo_mpc, mean_error_ppo_mpc + s * std_error_ppo_mpc, color=plot_colors['PPO-MPC'], alpha=0.2)
    ax.legend(ncol=1)
elif plot_name == 'Control-oriented':
    ax.plot(time_axis, mean_error_pid, color=plot_colors['Geometric Control'], label='Geometric Control')
    ax.plot(time_axis, mean_error_lqr, color=plot_colors['LQR'], label='LQR')
    ax.plot(time_axis, mean_error_ilqr, color=plot_colors['iLQR'], label='iLQR')
    ax.plot(time_axis, mean_error_lmpc, color=plot_colors['Linear MPC'], label='Linear MPC')
    ax.plot(time_axis, mean_error_mpc, color=plot_colors['Nonlinear MPC'], label='Nonlinear MPC')
    ax.plot(time_axis, mean_error_fmpc, color=plot_colors['F-MPC'], label='F-MPC')
    ax.plot(time_axis, mean_error_gpmpc, color=plot_colors['GP-MPC'], label='GP-MPC')
    if plot_std_tracking_error:
        ax.fill_between(time_axis, mean_error_pid - s * std_error_pid, mean_error_pid + s * std_error_pid, color=plot_colors['Geometric Control'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_lqr - s * std_error_lqr, mean_error_lqr + s * std_error_lqr, color=plot_colors['LQR'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_ilqr - s * std_error_ilqr, mean_error_ilqr + s * std_error_ilqr, color=plot_colors['iLQR'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_lmpc - s * std_error_lmpc, mean_error_lmpc + s * std_error_lmpc, color=plot_colors['Linear MPC'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_mpc - s * std_error_mpc, mean_error_mpc + s * std_error_mpc, color=plot_colors['Nonlinear MPC'], alpha=0.2)
        ax.fill_between(time_axis, mean_error_gpmpc - s * std_error_gpmpc, mean_error_gpmpc + s * std_error_gpmpc, color=plot_colors['GP-MPC'], alpha=0.2)
    ax.legend(ncol=2)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Tracking error [m]')
ax.set_ylim(-0.04, 0.3)

if additional == '11':
    fig.suptitle(f'Tracking error ({plot_name})',)
else:
    if additional == '9':
        fig.suptitle(f'Tracking error (faster) ({plot_name})',)
    elif additional == '15':
        fig.suptitle(f'Tracking error (slower) ({plot_name})',)

fig.tight_layout()

# Ensure directories exist
error_dir = script_path / 'error'
path_dir = script_path / 'path'
error_dir.mkdir(exist_ok=True)
path_dir.mkdir(exist_ok=True)

# Save error plot
file_name = f'tracking_error_{plot_name}_{additional}'
if not generalization:
    fig.savefig(error_dir / f'{file_name}.pdf', bbox_inches='tight')
    print(f'Saved at {error_dir / f"{file_name}.pdf"}')
    fig.savefig(error_dir / f'{file_name}.png', bbox_inches='tight')
    print(f'Saved at {error_dir / f"{file_name}.png"}')
else:
    fig.savefig(error_dir / f'{file_name}.pdf', bbox_inches='tight')
    print(f'Saved at {error_dir / f"{file_name}.pdf"}')
    fig.savefig(error_dir / f'{file_name}.png', bbox_inches='tight')
    print(f'Saved at {error_dir / f"{file_name}.png"}')

##################################################
# Plot the state path x, z [0, 2]
title_fontsize = 20
legend_fontsize = 12
axis_label_fontsize = 12
axis_tick_fontsize = 12
dummy = int(X_GOAL.shape[0] / 2)
fig, ax = plt.subplots(figsize=(8, 4))
# adjust the distance between title and the plot
fig.subplots_adjust(top=0.2)

# Plot the convex hull of each steps
hull_alpha = 0.3
plot_second_half = True  # Option to plot only the second half of the trajectory
max_steps = eval(additional) * STEPS_PER_SECOND
reference_idx = np.arange(0, max_steps, 1)
if plot_second_half:
    reference_idx = reference_idx[int(max_steps / 2):]

if plot_name == 'RL':
    plot_xz_trajectory_with_hull(ax, sac_traj_data, label='SAC',
                                 traj_color=plot_colors['SAC'], hull_color=plot_colors['SAC'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
    plot_xz_trajectory_with_hull(ax, ppo_traj_data, label='PPO',
                                 traj_color=plot_colors['PPO'], hull_color=plot_colors['PPO'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
    plot_xz_trajectory_with_hull(ax, dppo_traj_data, label='DPPO',
                                 traj_color=plot_colors['DPPO'], hull_color=plot_colors['DPPO'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
    plot_xz_trajectory_with_hull(ax, ppo_mpc_traj_data, label='PPO-MPC',
                                 traj_color=plot_colors['PPO-MPC'], hull_color=plot_colors['PPO-MPC'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
elif plot_name == 'Control-oriented':
    plot_xz_trajectory_with_hull(ax, pid_traj_data, label='Geometric Control',
                                 traj_color=plot_colors['Geometric Control'], hull_color=plot_colors['Geometric Control'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
    plot_xz_trajectory_with_hull(ax, lqr_traj_data, label='LQR',
                                 traj_color=plot_colors['LQR'], hull_color=plot_colors['LQR'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
    plot_xz_trajectory_with_hull(ax, ilqr_traj_data, label='iLQR',
                                 traj_color=plot_colors['iLQR'], hull_color=plot_colors['iLQR'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
    plot_xz_trajectory_with_hull(ax, gpmpc_traj_data, label='GP-MPC',
                                 traj_color=plot_colors['GP-MPC'], hull_color=plot_colors['GP-MPC'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
    plot_xz_trajectory_with_hull(ax, lmpc_traj_data, label='Linear MPC',
                                 traj_color=plot_colors['Linear MPC'], hull_color=plot_colors['Linear MPC'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
    plot_xz_trajectory_with_hull(ax, mpc_traj_data, label='Nonlinear MPC',
                                 traj_color=plot_colors['Nonlinear MPC'], hull_color=plot_colors['Nonlinear MPC'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)
    plot_xz_trajectory_with_hull(ax, fmpc_traj_data, label='F-MPC',
                                 traj_color=plot_colors['F-MPC'], hull_color=plot_colors['F-MPC'],
                                 alpha=hull_alpha, plot_second_half=plot_second_half)

ax.plot(X_GOAL[reference_idx, 0], X_GOAL[reference_idx, 2], color=plot_colors['Reference'], linestyle='-.', linewidth=1., label='Reference')
# ax.plot()
ax.set_xlabel('$x$ [m]', fontsize=axis_label_fontsize)
ax.set_ylabel('$z$ [m]', fontsize=axis_label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=axis_tick_fontsize)
# set the super title
if additional == '11':
    fig.suptitle(f'Evaluation ({plot_name})', fontsize=title_fontsize)
else:
    if additional == '9':
        fig.suptitle(f'Generalization (faster) ({plot_name} )', fontsize=title_fontsize)
    elif additional == '15':
        fig.suptitle(f'Generalization (slower) ({plot_name} )', fontsize=title_fontsize)
ax.set_ylim(0.35, 1.85)
ax.set_xlim(-1.6, 1.6)
fig.tight_layout()

# get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# specify order of items in legend
# order = [3, 1, 2, 0]
# order = [0, 4, 6, 1, 5, 2, 3]
order = np.arange(len(labels))

# add legend to plot
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=3, loc='upper center', fontsize=legend_fontsize)

# Save path plot
if additional == '11':
    fig.savefig(path_dir / f'{plot_name}_xz_path_performance_{additional}.pdf', bbox_inches='tight')
    print(f'Saved at {path_dir / f"{plot_name}_xz_path_performance_{additional}.pdf"}')
    fig.savefig(path_dir / f'{plot_name}_xz_path_performance_{additional}.png', bbox_inches='tight')
    print(f'Saved at {path_dir / f"{plot_name}_xz_path_performance_{additional}.png"}')
else:
    fig.savefig(path_dir / f'{plot_name}_xz_path_generalization_{additional}.pdf', bbox_inches='tight')
    print(f'Saved at {path_dir / f"{plot_name}_xz_path_generalization_{additional}.pdf"}')
    fig.savefig(path_dir / f'{plot_name}_xz_path_generalization_{additional}.png', bbox_inches='tight')
    print(f'Saved at {path_dir / f"{plot_name}_xz_path_generalization_{additional}.png"}')
