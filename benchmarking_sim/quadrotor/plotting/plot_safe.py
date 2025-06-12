import os
import sys

import munch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from safe_control_gym.utils.configuration import ConfigFactory
from functools import partial
from safe_control_gym.utils.registration import make
from benchmarking_sim.quadrotor.benchmark_util.utils import load_gym_data
from benchmarking_sim.quadrotor.plotting.safe.plot_helper_safe import (
    plot_xz_trajectory_with_hull,
    collect_state_constraint_values,
    plot_violations_over_time,
    plot_min_distance_to_boundary,
    # plot_constraint_value_distribution, # This was in the original import but not defined/used yet
    plot_constraint_violation_summary_boxplot,
    plot_constraint_violation_summary_violinplot
)

# get the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
script_path = os.path.dirname(os.path.realpath(__file__))
os.mkdir(f'{script_path}/safe') \
    if not os.path.exists(f'{script_path}/safe') else None

ref_color = 'black'
gpmpc_color = 'royalblue'
gpmpc_hull_color = 'cornflowerblue'
mpc_color = 'cadetblue'
mpc_hull_color = 'cadetblue'
ppo_color = 'darkorange'
ppo_hull_color = 'moccasin'

plot_colors = {
    'GP-MPC': 'royalblue',
    'PPO': 'darkorange',
    'SAC': 'red',
    'DPPO': 'pink',
    'PPO-MPC': 'tan',
    'PID': 'darkgray',
    'Linear MPC': 'green',
    'Nonlinear MPC': 'cadetblue',
    'iLQR': 'slateblue',
    'LQR': 'blueviolet',
    'F-MPC': 'darkblue',
    'MAX': 'none',
    'MIN': 'none',
}

episode_len = int(sys.argv[1]) if len(sys.argv) > 1 else 11

additional = ''

# get the config
ALGO = 'pid'
SYS = 'quadrotor_2D_attitude'
TASK = 'tracking'
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
nx = X_GOAL.shape[1]
X_GOAL = X_GOAL[:661, :]  # limit the goal to 661 steps for plotting
random_env.close()
nx = X_GOAL.shape[1]
dt = 1/60  # 60 Hz
##########################################################################
# from benchmarking_sim.quadrotor.extract_results import plot_trajectory
# additional = '_safety'
# algo = 'mpc_acados'

##########################################################################
# mpc_data = f'{script_path}/data/safe/no_noise.pkl'
# gpmpc_data = f'{script_path}/../gpmpc_acados_TP/results/safety_safety/temp/seed1_May-06-23-10-49_0d011d1/gpmpc_acados_TP_data_quadrotor_traj_tracking.pkl'

mpsf_data = f'{script_path}/../data/safe/no_noise.pkl'
# mpc_data = f'{script_path}/../data/traj_results_mpc_acados_quadrotor_2D_attitude_safety_noiseless.npy'
# gpmpc_data = f'{script_path}/../data/traj_results_gpmpc_acados_TP_quadrotor_2D_attitude_safety_noiseless.npy'
# ppo_mpc_data = f'{script_path}/../data/safe/traj_results_ppo_mpc_11_noiseless.npy'

# Load the data
mpsf_data = np.load(mpsf_data, allow_pickle=True)
uncert_mpsf_traj_data = np.array(mpsf_data['uncert_results']['obs'])[:,:,:nx]
cert_mpsf_traj_data = np.array(mpsf_data['cert_results']['obs'])[:,:,:nx]

# a constraint value example
# constraint_value = mpsf_data['cert_results']['info'][i][j]['constraint_values']
# collect all the constraint values in a list 
# the order of the element should be (episode, step, constraint_value_dim)

# constraint value > 0 means violation
# the constraint values are ordered as follows:
# [state_lower, state_upper, input_lower, input_upper]

cert_mpsf_constraint_values = collect_state_constraint_values(mpsf_data['cert_results'], nx)
uncert_mpsf_constraint_values = collect_state_constraint_values(mpsf_data['uncert_results'], nx)
print(f'cert_mpsf_constraint_values shape: {cert_mpsf_constraint_values.shape}')
print(f'uncert_mpsf_constraint_values shape: {uncert_mpsf_constraint_values.shape}')

# Calculate summed positive violations for ALL state constraints
sum_positive_violations_cert_all = None
if cert_mpsf_constraint_values.ndim == 3 and cert_mpsf_constraint_values.shape[0] > 0 and cert_mpsf_constraint_values.shape[1] > 0 and cert_mpsf_constraint_values.shape[2] > 0:
    sum_positive_violations_cert_all = np.sum(np.maximum(0, cert_mpsf_constraint_values), axis=2)
elif cert_mpsf_constraint_values.ndim == 2 and cert_mpsf_constraint_values.shape[0] > 0 and cert_mpsf_constraint_values.shape[1] > 0: # Already summed or single constraint
    sum_positive_violations_cert_all = np.maximum(0, cert_mpsf_constraint_values)

sum_positive_violations_uncert_all = None
if uncert_mpsf_constraint_values.ndim == 3 and uncert_mpsf_constraint_values.shape[0] > 0 and uncert_mpsf_constraint_values.shape[1] > 0 and uncert_mpsf_constraint_values.shape[2] > 0:
    sum_positive_violations_uncert_all = np.sum(np.maximum(0, uncert_mpsf_constraint_values), axis=2)
elif uncert_mpsf_constraint_values.ndim == 2 and uncert_mpsf_constraint_values.shape[0] > 0 and uncert_mpsf_constraint_values.shape[1] > 0: # Already summed or single constraint
    sum_positive_violations_uncert_all = np.maximum(0, uncert_mpsf_constraint_values)

# plot the constraint violations over time for all state constraints
fig_violations, ax_violations = plt.subplots(figsize=(10, 3))

if sum_positive_violations_cert_all is not None:
    plot_violations_over_time(ax_violations, 
                              sum_positive_violations_cert_all, 
                              dt, label='PPO+MPSF (All State Constraints)', color=plot_colors['Linear MPC'])
if sum_positive_violations_uncert_all is not None:
    plot_violations_over_time(ax_violations,
                              sum_positive_violations_uncert_all, 
                              dt, label='PPO (All State Constraints)', color=plot_colors['PPO'])

ax_violations.set_xlabel('Time (s)')
ax_violations.set_ylabel('State Constraint violation Magnitude')
ax_violations.set_title('State Constraint Violation Magnitude Over Time')
ax_violations.legend(ncol=2, loc='best')
ax_violations.grid(True)
fig_violations.tight_layout()
violation_plot_path = f'{script_path}/safe/quadrotor_all_state_constraint_violations.png'
fig_violations.savefig(violation_plot_path, dpi=300)
print(f'Saved all state constraint violation plot to {violation_plot_path}')

# plot the mean and std of the position constraint violations over time
# position constraint violations have index 0, 2, 6, 8 in the constraint values
position_constraint_indices = [0, 2, 6, 8] # x_lower, x_upper, z_lower, z_upper (example interpretation)

cert_pos_constraint_values = collect_state_constraint_values(mpsf_data['cert_results'], nx, constraint_indices=position_constraint_indices)
uncert_pos_constraint_values = collect_state_constraint_values(mpsf_data['uncert_results'], nx, constraint_indices=position_constraint_indices)

print(f'cert_pos_constraint_values shape: {cert_pos_constraint_values.shape}')
print(f'uncert_pos_constraint_values shape: {uncert_pos_constraint_values.shape}')

# Calculate summed positive violations for POSITIONAL constraints
sum_pos_violations_cert = None
if cert_pos_constraint_values.ndim == 3 and cert_pos_constraint_values.shape[0] > 0 and cert_pos_constraint_values.shape[1] > 0 and cert_pos_constraint_values.shape[2] > 0:
    sum_pos_violations_cert = np.sum(np.maximum(0, cert_pos_constraint_values), axis=2)
elif cert_pos_constraint_values.ndim == 2 and cert_pos_constraint_values.shape[0] > 0 and cert_pos_constraint_values.shape[1] > 0:
    sum_pos_violations_cert = np.maximum(0, cert_pos_constraint_values)

sum_pos_violations_uncert = None
if uncert_pos_constraint_values.ndim == 3 and uncert_pos_constraint_values.shape[0] > 0 and uncert_pos_constraint_values.shape[1] > 0 and uncert_pos_constraint_values.shape[2] > 0:
    sum_pos_violations_uncert = np.sum(np.maximum(0, uncert_pos_constraint_values), axis=2)
elif uncert_pos_constraint_values.ndim == 2 and uncert_pos_constraint_values.shape[0] > 0 and uncert_pos_constraint_values.shape[1] > 0:
    sum_pos_violations_uncert = np.maximum(0, uncert_pos_constraint_values)


fig_pos_violations, ax_pos_violations = plt.subplots(figsize=(10, 3))

if sum_pos_violations_cert is not None:
    plot_violations_over_time(ax_pos_violations,
                              sum_pos_violations_cert,
                              dt, label='PPO+MPSF (Positional Constraints)', color=plot_colors['Linear MPC'])
if sum_pos_violations_uncert is not None:
    plot_violations_over_time(ax_pos_violations,
                              sum_pos_violations_uncert,
                              dt, label='PPO (Positional Constraints)', color=plot_colors['PPO'])

ax_pos_violations.set_xlabel('Time (s)')
ax_pos_violations.set_ylabel('Positional Constraint Violation [m]') # Assuming positions are in meters
ax_pos_violations.set_title('Positional Constraint Violation Magnitude Over Time')
ax_pos_violations.legend(ncol=2, loc='best')
ax_pos_violations.grid(True)
fig_pos_violations.tight_layout()
pos_violation_plot_path = f'{script_path}/safe/quadrotor_positional_constraint_violations.png'
fig_pos_violations.savefig(pos_violation_plot_path, dpi=300)
print(f'Saved positional constraint violation plot to {pos_violation_plot_path}')

# Plot the minimum distance to positional boundaries
fig_min_dist, ax_min_dist = plt.subplots(figsize=(10, 3))

# Use the already collected cert_pos_constraint_values and uncert_pos_constraint_values
# These arrays have shape (num_episodes, num_steps, num_positional_constraints)
if cert_pos_constraint_values.size > 0:
    plot_min_distance_to_boundary(ax_min_dist,
                                  cert_pos_constraint_values,
                                  dt,
                                  label='PPO+MPSF',
                                  color=plot_colors['Linear MPC'])
if uncert_pos_constraint_values.size > 0:
    plot_min_distance_to_boundary(ax_min_dist,
                                  uncert_pos_constraint_values,
                                  dt,
                                  label='PPO',
                                  color=plot_colors['PPO'])

ax_min_dist.set_xlabel('Time (s)')
ax_min_dist.set_ylabel('Min. Distance to Pos. Boundary [m]')
ax_min_dist.set_title('Min. Distance to Positional Safety Boundary Over Time')

# Ensure legend handles multiple lines for y=0 if called multiple times
handles, labels = ax_min_dist.get_legend_handles_labels()
if handles: # Check if there are any handles to prevent error if no data was plotted
    by_label = dict(zip(labels, handles)) # Remove duplicate labels for axhline
    ax_min_dist.legend(by_label.values(), by_label.keys())

ax_min_dist.grid(True)
fig_min_dist.tight_layout()
min_dist_plot_path = f'{script_path}/safe/quadrotor_min_dist_to_boundary.png'
fig_min_dist.savefig(min_dist_plot_path, dpi=300)
print(f'Saved minimum distance to boundary plot to {min_dist_plot_path}')

# Create box plot for summary of positional constraint violations
fig_pos_constraints_boxplot, ax_pos_constraints_boxplot = plt.subplots(figsize=(8, 4))
all_pos_constraint_values_for_boxplot = {}

if cert_pos_constraint_values is not None and cert_pos_constraint_values.size > 0 :
    all_pos_constraint_values_for_boxplot['PPO+MPSF'] = cert_pos_constraint_values
if uncert_pos_constraint_values is not None and uncert_pos_constraint_values.size > 0:
    all_pos_constraint_values_for_boxplot['PPO'] = uncert_pos_constraint_values

# Define colors for the box plot controllers based on existing plot_colors
boxplot_controller_colors = {
    'PPO+MPSF': plot_colors.get('Linear MPC', 'gray'), 
    'PPO': plot_colors.get('PPO', 'gray')
}

if all_pos_constraint_values_for_boxplot:
    plot_constraint_violation_summary_boxplot(ax_pos_constraints_boxplot, 
                                              all_pos_constraint_values_for_boxplot, 
                                              boxplot_controller_colors)
    ax_pos_constraints_boxplot.set_title('Distribution of Positional Constraint Values')
    fig_pos_constraints_boxplot.tight_layout()
    pos_constraints_boxplot_path = f'{script_path}/safe/quadrotor_positional_constraint_values_boxplot.png'
    fig_pos_constraints_boxplot.savefig(pos_constraints_boxplot_path, dpi=300)
    print(f'Saved positional constraint values boxplot to {pos_constraints_boxplot_path}')
else:
    print("Skipping positional constraint values box plot generation as no data was available.")

# Create violin plot for summary of positional constraint violations
fig_pos_constraints_violinplot, ax_pos_constraints_violinplot = plt.subplots(figsize=(8, 4))
# all_pos_constraint_values_for_boxplot is already defined and populated from the boxplot section
# Reusing boxplot_controller_colors for consistency, or define new ones if needed
violinplot_controller_colors = boxplot_controller_colors 

if all_pos_constraint_values_for_boxplot: # Check if data exists
    plot_constraint_violation_summary_violinplot(ax_pos_constraints_violinplot,
                                                 all_pos_constraint_values_for_boxplot,
                                                 violinplot_controller_colors)
    # ax_pos_constraints_violinplot.set_title('Distribution of Positional Constraint Values (Violin Plot)') # Title is set in helper
    # ax_pos_constraints_violinplot.set_ylabel('Constraint Value') # Y-label is set in helper
    # ax_pos_constraints_violinplot.grid(True, axis='y') # Grid is set in helper
    fig_pos_constraints_violinplot.tight_layout()
    pos_constraints_violinplot_path = f'{script_path}/safe/quadrotor_positional_constraint_values_violinplot.png'
    fig_pos_constraints_violinplot.savefig(pos_constraints_violinplot_path, dpi=300)
    print(f'Saved positional constraint values violin plot to {pos_constraints_violinplot_path}')
else:
    print("Skipping positional constraint values violin plot generation as no data was available.")


# mpc_data = np.load(mpc_data, allow_pickle=True)
# gpmpc_data = np.load(gpmpc_data, allow_pickle=True)
# ppo_mpc_data = np.load(ppo_mpc_data, allow_pickle=True).item()
# ppo_mpc_traj_data = np.array(ppo_mpc_data['obs'])[:, :,:nx]
# mpc_traj_data = load_gym_data(mpc_data)
# gpmpc_traj_data = load_gym_data(gpmpc_data)

# total_steps = mpc_traj_data['action'].shape[0]
# time_axis = np.arange(0, mpc_traj_data['action'].shape[0]) * 1/60

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

ax.plot(X_GOAL[:, 0], X_GOAL[:, 2], label='Reference', 
        color=ref_color, linestyle='dashdot', linewidth=2.0)
# ax.plot(mpc_data[0][:, 0], mpc_data[0][:, 2],
#         label='MPC', color=mpc_color, linewidth=2.0)
# plot_xz_trajectory_with_hull(ax, mpc_data, label='MPC',
#                                 traj_color=mpc_color, hull_color=mpc_hull_color,
#                                 linewidth=2.0, alpha=alpha, padding_factor=k)
# plot_xz_trajectory_with_hull(ax, gpmpc_data, label='GP-MPC',
#                                 traj_color=gpmpc_color, hull_color=gpmpc_hull_color,
#                                 linewidth=2.0, alpha=alpha, padding_factor=k)
plot_xz_trajectory_with_hull(ax, cert_mpsf_traj_data, label='PPO+MPSF',
                                traj_color=plot_colors['Linear MPC'], hull_color=plot_colors['DPPO'],
                                linewidth=2.0, alpha=alpha, padding_factor=k)
# plot_xz_trajectory_with_hull(ax, ppo_mpc_traj_data, label='PPO-MPC',
#                             traj_color=plot_colors['PPO-MPC'], hull_color=plot_colors['PPO'],
#                             linewidth=2.0, alpha=alpha, padding_factor=k)
plot_xz_trajectory_with_hull(ax, uncert_mpsf_traj_data, label='PPO',
                                traj_color=plot_colors['PPO'], hull_color=plot_colors['PPO'],
                                linewidth=2.0, alpha=alpha, padding_factor=k)
# plot x-z path trajectory
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(mpc_traj_data['ref'][:, 0], mpc_traj_data['ref'][:, 2],
#             label='Reference', color='gray', linestyle='dashdot')
# ax.plot(mpc_traj_data['obs'][:, 0], mpc_traj_data['obs'][:, 2], 
#             label='MPC', color='aqua', linewidth=2)
# ax.plot(gpmpc_traj_data['obs'][:, 0], gpmpc_traj_data['obs'][:, 2],
#             label='GP-MPC', color='royalblue', linewidth=2)

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


ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_title('Constrained Trajectory Tracking')
  
fig.tight_layout()
fig.savefig(f'{script_path}/safe/quadrotor_traj_tracking_xz.png', dpi=300)
print(f'Saved figure to {script_path}/safe/quadrotor_traj_tracking_xz.png')

# save a zoomed in version of the plot focusing on the left boundary
plt.xlim(-1.1, -0.75)
plt.ylim(0.6, 1.0)
fig.tight_layout()
fig.savefig(f'{script_path}/safe/quadrotor_traj_tracking_xz_zoomed_left_boundary.png', dpi=300)
print(f'Saved zoomed in figure to {script_path}/safe/quadrotor_traj_tracking_xz_zoomed_left_boundary.png')

# save a zoomed in version of the plot focusing on the upper boundary
plt.xlim(-1.1, -0.4)
plt.ylim(1.2, 1.55)
fig.tight_layout()
fig.savefig(f'{script_path}/safe/quadrotor_traj_tracking_xz_zoomed_upper_boundary.png', dpi=300)
print(f'Saved zoomed in figure to {script_path}/safe/quadrotor_traj_tracking_xz_zoomed_upper_boundary.png')