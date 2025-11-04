
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config

ALGO = 'mpc_acados'
SYS = 'quadrotor_2D_attitude'
TASK = 'tracking'
ADDITIONAL = ''
PRIOR = '100'
agent = 'quadrotor' if SYS in ['quadrotor_2D', 'quadrotor_2D_attitude', 'quadrotor_3D_attitude'] else SYS

# Check if the config file exists
assert os.path.exists(f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml'), f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml does not exist'
assert os.path.exists(f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'

sys.argv[1:] = ['--algo', ALGO,
                '--task', agent,
                '--overrides',
                f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml',
                f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                '--seed', '1',
                '--use_gpu', 'True',
                '--output_dir', f'./{ALGO}/results',
                ]

fac = ConfigFactory()
fac.add_argument('--func', type=str, default='train', help='main function to run.')
fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
# Merge config and create output directory
config = fac.merge()
if ALGO in ['gpmpc_acados', 'gp_mpc', 'gpmpc_acados_TP']:
    num_data_max = config.algo_config.num_epochs * config.algo_config.num_samples
    config.output_dir = os.path.join(config.output_dir, PRIOR + '_' + repr(num_data_max))
set_dir_from_config(config)
config.algo_config.output_dir = config.output_dir
mkdirs(config.output_dir)

# Create an environment
env_func = partial(make,
                   config.task,
                   seed=config.seed,
                   **config.task_config
                   )
env = env_func(gui=False)


###########################################################
# Load the data
data_folder_dir = './data/'

ilqr_data = np.load(data_folder_dir + 'ilqr_ref_traj.npy', allow_pickle=True).item()
mpc_constrained_data = np.load(data_folder_dir + 'mpc_ref_traj_constrained.npy', allow_pickle=True).item()
mpc_unconstrained_data = np.load(data_folder_dir + 'mpc_ref_traj_unconstrained.npy', allow_pickle=True).item()

ilqr_state_traj = ilqr_data['obs'][0]
mpc_state_constrained_traj = mpc_constrained_data['obs'][0]
mpc_state_unconstrained_traj = mpc_unconstrained_data['obs'][0]

ilqr_action_traj = ilqr_data['action'][0]
mpc_action_constrained_traj = mpc_constrained_data['action'][0]
mpc_action_unconstrained_traj = mpc_unconstrained_data['action'][0]

# Old ilqr reference
ilqr_old_data_dir = '/home/mingxuan/Repositories/scg_tsung/examples/lqr/ilqr_ref_traj.npy'
ilqr_old_data = np.load(ilqr_old_data_dir, allow_pickle=True).item()
ilqr_old_state_traj = ilqr_old_data['obs'][0]
ilqr_old_action_traj = ilqr_old_data['action'][0]

rmse_mpc_c = 0.054
rmse_mpc_uc = 0.054
rmse_ilqr = 0.052

print('RMSE MPC Constrained:', rmse_mpc_c)
print('RMSE MPC Unconstrained:', rmse_mpc_uc)
print('RMSE iLQR:', rmse_ilqr)

model = env.symbolic
stepsize = model.dt
plot_length = np.min([np.shape(ilqr_action_traj)[0], np.shape(ilqr_state_traj)[0]])
times = np.linspace(0, stepsize * plot_length, plot_length)

reference = env.X_GOAL
if env.TASK == Task.STABILIZATION:
    reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

nx = model.nx
nu = model.nu
# Plot state traj
fig, axs = plt.subplots(model.nx, figsize=(8, model.nx * 1))
for k in range(model.nx):
    axs[k].plot(times, np.array(ilqr_state_traj).transpose()[k, 0:plot_length], label='iLQR')
    axs[k].plot(times, np.array(mpc_state_constrained_traj).transpose()[k, 0:plot_length], label='MPC Constrained', linestyle='-.')
    axs[k].plot(times, np.array(mpc_state_unconstrained_traj).transpose()[k, 0:plot_length], label='MPC Unconstrained', linestyle='--')
    axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
    axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
    axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if k != model.nx - 1:
        axs[k].set_xticks([])
axs[0].set_title('State Trajectories')
axs[-1].legend(ncol=4, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
axs[-1].set(xlabel='time (sec)')
fig.tight_layout()
fig.savefig('./data/state_trajectories.png')

# Plot action traj
env_action_bound_high = env.action_space.high
env_action_bound_low = env.action_space.low
print('Action bounds:', env_action_bound_low, env_action_bound_high)


fig, axs = plt.subplots(model.nu, figsize=(8, model.nu * 2))
if model.nu == 1:
    axs = [axs]
for k in range(model.nu):
    axs[k].plot(times, np.array(ilqr_action_traj).transpose()[k, 0:plot_length], label='iLQR')
    axs[k].plot(times, np.array(mpc_action_constrained_traj).transpose()[k, 0:plot_length], label='MPC Constrained', linestyle='-.')
    axs[k].plot(times, np.array(mpc_action_unconstrained_traj).transpose()[k, 0:plot_length], label='MPC Unconstrained', linestyle='--')
    axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
    axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[k].plot(times, np.ones_like(times) * env_action_bound_high[k], color='k', linestyle='--', label='action bounds')
    axs[k].plot(times, np.ones_like(times) * env_action_bound_low[k], color='k', linestyle='--')
axs[0].set_title('Input Trajectories')
axs[-1].set(xlabel='time (sec)')
axs[-1].legend(ncol=4, loc='best')
fig.tight_layout()
fig.savefig('./data/input_trajectories.png')

# Plot the figure-eight
x_idx, z_idx = 0, 2
fig, axs = plt.subplots(1)
axs.plot(np.array(ilqr_state_traj).transpose()[x_idx, 0:plot_length],
         np.array(ilqr_state_traj).transpose()[z_idx, 0:plot_length], label='iLQR')
axs.plot(np.array(mpc_state_constrained_traj).transpose()[x_idx, 0:plot_length],
         np.array(mpc_state_constrained_traj).transpose()[z_idx, 0:plot_length], label='MPC Constrained', linestyle='-.')
axs.plot(np.array(mpc_state_unconstrained_traj).transpose()[x_idx, 0:plot_length],
         np.array(mpc_state_unconstrained_traj).transpose()[z_idx, 0:plot_length], label='MPC Unconstrained', linestyle='--')
axs.plot(reference.transpose()[x_idx, 0:plot_length],
         reference.transpose()[z_idx, 0:plot_length], color='r', label='desired')
axs.set_xlabel('x [m]')
axs.set_ylabel('z [m]')
axs.set_title('State path in x-z plane')
axs.legend()
fig.tight_layout()
fig.savefig('./data/state_xz_path.png')


# Compare ilqr and ilqr_old
fig, axs = plt.subplots(1)
axs.plot(np.array(ilqr_state_traj).transpose()[x_idx, 0:plot_length],
         np.array(ilqr_state_traj).transpose()[z_idx, 0:plot_length], label='iLQR')
axs.plot(np.array(ilqr_old_state_traj).transpose()[x_idx, 0:plot_length],
         np.array(ilqr_old_state_traj).transpose()[z_idx, 0:plot_length], label='iLQR old', linestyle='-.', color='gray')
axs.plot(reference.transpose()[x_idx, 0:plot_length],
         reference.transpose()[z_idx, 0:plot_length], color='r', label='desired')
axs.set_xlabel('x [m]')
axs.set_ylabel('z [m]')
axs.set_title('State path in x-z plane')
axs.legend()
fig.tight_layout()
fig.savefig('./data/state_xz_path_ilqr_old.png')


# Calculate the accleeration and jerk of the full state trajectory
ilqr_acc_traj = np.diff(ilqr_state_traj, axis=0) / stepsize
ilqr_jerk_traj = np.diff(ilqr_acc_traj, axis=0) / stepsize

mpc_acc_constrained_traj = np.diff(mpc_state_constrained_traj, axis=0) / stepsize
mpc_jerk_constrained_traj = np.diff(mpc_acc_constrained_traj, axis=0) / stepsize

mpc_acc_unconstrained_traj = np.diff(mpc_state_unconstrained_traj, axis=0) / stepsize
mpc_jerk_unconstrained_traj = np.diff(mpc_acc_unconstrained_traj, axis=0) / stepsize

ref_acc_traj = np.diff(reference, axis=0) / stepsize
ref_jerk_traj = np.diff(ref_acc_traj, axis=0) / stepsize

z_acc_bound = np.array([-0.7 * 9.81, 0.8 * 9.81])

# Plot the acceleration
fig, axs = plt.subplots(3, figsize=(8, model.nx * 1))
acc_label = ['x_ddot [$m/s^2$]', 'z_ddot [$m/s^2$]', 'theta_ddot [$rad/s^2$]']
jerk_label = ['x_dddot [$m/s^3$]', 'z_dddot [$m/s^3$]', 'theta_dddot [$rad/s^3$]']
for idx, k in enumerate([1, 3, 5]):
    axs[idx].plot(times[1:], np.array(ilqr_acc_traj).transpose()[k, 0:plot_length - 1], label='iLQR')
    axs[idx].plot(times[1:], np.array(ref_acc_traj).transpose()[k, 0:plot_length - 1], color='r', label='fig 8')
    axs[idx].set(ylabel=acc_label[int(k // 2)])
    axs[idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if idx == 1:
        axs[idx].plot(times[1:], np.ones_like(times[1:]) * z_acc_bound[0], color='k', linestyle='--', label='acceleration bounds')
        axs[idx].plot(times[1:], np.ones_like(times[1:]) * z_acc_bound[1], color='k', linestyle='--')
axs[0].set_title('Acceleration Trajectories')
axs[-1].legend(ncol=4, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
axs[-1].set(xlabel='time (sec)')
fig.tight_layout()
fig.savefig('./data/acceleration_trajectories.png')

# Plot the jerk
fig, axs = plt.subplots(3, figsize=(8, model.nx * 1))
for idx, k in enumerate([1, 3, 5]):
    axs[idx].plot(times[2:], np.array(ilqr_jerk_traj).transpose()[k, 0:plot_length - 2], label='iLQR')
    axs[idx].plot(times[2:], np.array(ref_jerk_traj).transpose()[k, 0:plot_length - 2], color='r', label='fig 8')
    axs[idx].set(ylabel=jerk_label[int(k // 2)])
    axs[idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[0].set_title('Jerk Trajectories')
axs[-1].legend(ncol=4, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
axs[-1].set(xlabel='time (sec)')
fig.tight_layout()
fig.savefig('./data/jerk_trajectories.png')

env.close()
