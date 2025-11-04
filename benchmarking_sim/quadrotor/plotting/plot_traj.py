import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from benchmarking_sim.quadrotor.benchmark_util.utils import load_gym_data
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

# # get the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
script_path = os.path.dirname(os.path.realpath(__file__))

episode_len = int(sys.argv[1]) if len(sys.argv) > 1 else 11

additional = ''
# additional = '11'
# additional = '9'
# additional = '15'

# get the config
# ALGO = 'mpc_acados'
ALGO = 'pid'
SYS = 'quadrotor_2D_attitude'
# SYS = 'quadrotor_3D_attitude'
TASK = 'tracking'
# PRIOR = '200_hpo'
PRIOR = '100'
agent = 'quadrotor' if SYS in ['quadrotor_2D', 'quadrotor_2D_attitude', 'quadrotor_3D_attitude'] else SYS
SAFETY_FILTER = None

# check if the config file exists
assert os.path.exists(f'{script_path}/../config_overrides/{SYS}_{TASK}{additional}.yaml'), \
    f'{script_path}/../config_overrides/{SYS}_{TASK}{additional}.yaml does not exist'
assert os.path.exists(f'{script_path}/../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), \
    f'{script_path}/../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
if SAFETY_FILTER is None:
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', agent,
                    '--overrides',
                    f'{script_path}/../config_overrides/{SYS}_{TASK}{additional}.yaml',
                    f'{script_path}/../config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                    '--seed', '2',
                    '--use_gpu', 'True',
                    '--output_dir', f'./{ALGO}/results',
                    ]
fac = ConfigFactory()
fac.add_argument('--func', type=str, default='train', help='main function to run.')
fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
config = fac.merge()
# episode_len = config.task_config['episode_len_sec']
config.task_config['episode_len_sec'] = int(episode_len)
print(f'Episode length: {episode_len}')

# Create an environment
env_func = partial(make,
                   config.task,
                   seed=config.seed,
                   **config.task_config
                   )
random_env = env_func(gui=False)
X_GOAL = random_env.X_GOAL

# ref_type = 'ilqr'
ref_type = 'mpc'
# load the trajectory data
if ref_type == 'mpc':
    if episode_len == 9:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed2_Feb-07-12-04-42_82e2f47/mpc_acados_data_quadrotor_traj_tracking.pkl'
    if episode_len == 9:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-14-48-04_82e2f47/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-17-01-26_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 10:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-14-48-44_82e2f47/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-17-01-46_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 11:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-14-49-04_82e2f47/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-16-56-52_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 12:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-14-49-27_82e2f47/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-17-02-17_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 13:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-14-49-47_82e2f47/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-17-02-33_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 14:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-14-50-08_82e2f47/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-17-11-26_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 15:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-14-50-37_82e2f47/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-17-03-07_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
elif ref_type == 'ilqr':
    if episode_len == 9:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-19-21-35_1b67bc8/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-19-18-16_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 10:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-19-22-00_1b67bc8/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-19-18-42_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 11:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-19-22-22_1b67bc8/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-19-19-05_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 12:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-19-22-49_1b67bc8/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-19-19-24_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 13:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-19-23-14_1b67bc8/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-19-19-48_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 14:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-19-23-38_1b67bc8/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed1_Feb-07-19-35-57_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'
    elif episode_len == 15:
        mpc_data = f'{script_path}/../mpc_acados/results/temp/seed4_Feb-07-19-24-03_1b67bc8/mpc_acados_data_quadrotor_traj_tracking.pkl'
        ilqr_data = f'{script_path}/../ilqr/results/temp/seed4_Feb-07-19-20-43_1b67bc8/ilqr_data_quadrotor_traj_tracking.pkl'

mpc_traj_data = load_gym_data(mpc_data)
ilqr_traj_data = load_gym_data(ilqr_data)
total_steps = mpc_traj_data['action'].shape[0]
time_axis = np.arange(0, mpc_traj_data['action'].shape[0]) * 1 / 60

nx = 6
nu = 2
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(X_GOAL[:, 0], X_GOAL[:, 2],
           color='k', linestyle=':', label='Figure-8')
ax[0].plot(mpc_traj_data['ref'][:, 0], mpc_traj_data['ref'][:, 2],
           color='red', linestyle='--', label='Feasible Ref.')
ax[0].plot(mpc_traj_data['obs'][:, 0], mpc_traj_data['obs'][:, 2],
           color=colors[0], label='MPC')
ax[0].plot(ilqr_traj_data['obs'][:, 0], ilqr_traj_data['obs'][:, 2],
           color=colors[1], label='iLQR')
ax[0].set_xlabel('x [m]')
ax[0].set_ylabel('z [m]')
ax[0].legend()
ax[0].set_title(f'Trajectory path x-z \n MPC RMSE {mpc_traj_data["rmse"]:.3f} [m], iLQR RMSE {ilqr_traj_data["rmse"]:.3f} [m]')

# Plot tracking error
ax[1].plot(time_axis, mpc_traj_data['error'], color=colors[0])
ax[1].plot(time_axis, ilqr_traj_data['error'], color=colors[1])
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Tracking error [m]')
ax[1].set_title('Tracking error')
fig.tight_layout()
fig.savefig(os.path.join(script_path, f'{ref_type}_ref_traj_{episode_len}.png'), bbox_inches='tight')

print(f'length sec: {episode_len}, MPC RMSE: {mpc_traj_data["rmse"]:.3f}, iLQR RMSE: {ilqr_traj_data["rmse"]:.3f}')
