import argparse
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from benchmarking_sim.quadrotor.benchmark_util.utils import load_metric, plot_colors

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate animated radar plots for RL controllers')
parser.add_argument('--controller', type=str, choices=['PPO', 'SAC', 'DPPO', 'all'],
                    default='all', help='Controller to animate (default: all)')
parser.add_argument('--robustness-type', type=str, choices=['relative', 'abs'],
                    default='relative', help='Robustness type to use (default: relative)')
parser.add_argument('--fps', type=int, default=30, help='Animation frame rate (default: 30)')
parser.add_argument('--frames', type=int, default=180, help='Number of animation frames (default: 180)')
parser.add_argument('--hold-frames', type=int, default=60, help='Number of frames to hold at end (default: 60)')

args = parser.parse_args()

# Set up plotting style
seaborn.set_palette('deep')
seaborn.set_context('paper', font_scale=1.5)

plt.rcParams.update({
    'grid.alpha': 1.0,
    'savefig.bbox': 'tight',
})

script_dir = os.path.dirname(__file__)

# Animation settings from arguments
num_animation_frames = args.frames
frame_hold = args.hold_frames

# Configuration
robustness_type = args.robustness_type
save_folder = 'radar_animation'

print(f'Generating radar animation for: {args.controller}')
print(f'Robustness type: {robustness_type}')
print(f'Animation frames: {num_animation_frames}, Hold frames: {frame_hold}, FPS: {args.fps}')

# Load metrics
transfer_metric = {}
transfer_metric = load_metric(script_dir, transfer_metric, 'iLQR')
transfer_metric = load_metric(script_dir, transfer_metric, 'F-MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Nonlinear MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Linear MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Geometric Control')
transfer_metric = load_metric(script_dir, transfer_metric, 'LQR')
transfer_metric = load_metric(script_dir, transfer_metric, 'GP-MPC')

# Metric configuration
metric_index = {
    'worst_generalization_performance': 0,
    'performance': 1,
    'inference_time': 2,
    'model_complexity': 3,
    'sampling_complexity': 4,
    'robustness_proc': 5,
    'robustness_obs': 6,
    'robustness_param': 7,
}

inverted_axes_name = [
    'worst_generalization_performance',
    'performance',
    'inference_time',
    'sampling_complexity',
]
inverted_axes_index = [metric_index[i] for i in inverted_axes_name]

axis_legend_dict = {
    'worst_generalization_performance': 'Generalization',
    'performance': 'Performance',
    'inference_time': 'Online\nComputation',
    'model_complexity': 'Required Model\nKnowledge',
    'sampling_complexity': 'Sampling\ncomplexity',
    'robustness_proc': 'P',
    'robustness_obs': 'O',
    'robustness_param': r'$\theta$',
}

# ID numbers for specific methods
ID_numbers = {
    'robustness_obs': {
        'PPO': 140,
        'SAC': 110,
        'DPPO': 90,
    },
    'robustness_proc': {
        'PPO': 45,
        'SAC': 35,
        'DPPO': 35,
    },
    'robustness_param': {
        'PPO': 5.0,
        'SAC': 5,
        'DPPO': 5,
    },
    'worst_generalization_performance': {
        'PPO': 0.05,
        'SAC': 0.084,
        'DPPO': 0.07
    }
}

# Configuration parameters
axis_label_fontsize = 20
text_fontsize = 30
supertitle_fontsize = 30
subtitle_fontsize = 30
small_text_size = 20
padding = 0.15
OOD_alpha = 0.4
ID_alpha = 0.15


def normalize_data(data, max_values, min_values, inverted_axes_name, lower_padding):
    '''Normalize the data based on max and min values, applying padding and inversion where necessary.'''
    normalized_data = {}

    for key in data.keys():
        temp_value = (np.array(data[key]) - min_values[key]) / (max_values[key] - min_values[key])
        temp_value = (1 - temp_value) if key in inverted_axes_name else temp_value
        temp_value = np.clip(temp_value, 0, 1)  # clip to [0, 1]
        normalized_data[key] = temp_value + lower_padding

    return normalized_data


# Initialize metrics data
methods = [
    'GP-MPC', 'Linear MPC', 'Nonlinear MPC', 'F-MPC',
    'PPO', 'SAC', 'DPPO', 'PPO-MPC',
    'Geometric Control', 'iLQR', 'LQR'
]

metrics_data = {}
# Initialize the dictionary structure
for method in methods:
    metrics_data[method] = {
        'worst_generalization_performance': 0,
        'performance': 0,
        'inference_time': 0,
        'model_complexity': 0,
        'sampling_complexity': 0,
        'robustness_proc': 0,
        'robustness_obs': 0,
        'robustness_param': 0,
    }

# Fill in the metrics data (using the same data from plot_radar.py)
# GP-MPC
metrics_data['GP-MPC']['worst_generalization_performance'] = max(transfer_metric['GP-MPC']['rmse'][0], transfer_metric['GP-MPC']['rmse'][-1])
metrics_data['GP-MPC']['performance'] = transfer_metric['GP-MPC']['rmse'][2]
metrics_data['GP-MPC']['inference_time'] = transfer_metric['GP-MPC']['inference_time']
metrics_data['GP-MPC']['model_complexity'] = 1
metrics_data['GP-MPC']['sampling_complexity'] = 660
metrics_data['GP-MPC']['robustness_proc'] = 5
metrics_data['GP-MPC']['robustness_obs'] = 50
metrics_data['GP-MPC']['robustness_param'] = 4.5

# PPO
metrics_data['PPO']['worst_generalization_performance'] = max(0.1623562128527632, 0.1497925874584164)
metrics_data['PPO']['performance'] = 0.013743474079490085
metrics_data['PPO']['inference_time'] = 7.31e-5
metrics_data['PPO']['model_complexity'] = 3
metrics_data['PPO']['sampling_complexity'] = 382800
metrics_data['PPO']['robustness_obs'] = 14
metrics_data['PPO']['robustness_proc'] = 4
metrics_data['PPO']['robustness_param'] = 1.6

# SAC
metrics_data['SAC']['worst_generalization_performance'] = max(0.08555534895458854, 0.07275009863780316)
metrics_data['SAC']['performance'] = 0.015993880346779555
metrics_data['SAC']['inference_time'] = 8.72e-5
metrics_data['SAC']['model_complexity'] = 3
metrics_data['SAC']['sampling_complexity'] = 343200
metrics_data['SAC']['robustness_obs'] = 60
metrics_data['SAC']['robustness_proc'] = 4
metrics_data['SAC']['robustness_param'] = 1.8

# DPPO
metrics_data['DPPO']['worst_generalization_performance'] = max(0.14182353098764516, 0.1659762747503099)
metrics_data['DPPO']['performance'] = 0.022046533485966666
metrics_data['DPPO']['inference_time'] = 7.28e-5
metrics_data['DPPO']['model_complexity'] = 3
metrics_data['DPPO']['sampling_complexity'] = 1135200
metrics_data['DPPO']['robustness_obs'] = 20
metrics_data['DPPO']['robustness_proc'] = 6
metrics_data['DPPO']['robustness_param'] = 2.0

# Set max and min values for normalization
max_values = {key: max([metrics_data[method][key] for method in metrics_data.keys()]) for key in metrics_data['GP-MPC'].keys()}
min_values = {key: min([metrics_data[method][key] for method in metrics_data.keys()]) for key in metrics_data['GP-MPC'].keys()}

# Hand-tune max and min to make the plot look better
max_values['performance'] = 0.04
max_values['worst_generalization_performance'] = 0.087
max_values['sampling_complexity'] = 400000
max_values['robustness_proc'] = 10
max_values['robustness_obs'] = 100
max_values['robustness_param'] = 5.0
max_values['inference_time'] = 1.7e-3

# Set up radar plot angles
num_axis = len(metrics_data['GP-MPC'].keys())
categories = list(metrics_data['GP-MPC'].keys())
tiks = [axis_legend_dict.get(cat, cat) for cat in categories]

start_angle = np.deg2rad(0)
angles = [
    start_angle,  # worst generalization
    np.deg2rad(60),  # performance
    np.deg2rad(120),  # inference time
    np.deg2rad(180),  # model complexity
    np.deg2rad(240),  # sampling complexity
    np.deg2rad(300 - 15),  # robustness proc
    np.deg2rad(300),  # observation noise
    np.deg2rad(300 + 15),  # robustness param
]

# Add closing angle for plotting but not for labels
angles_closed = angles + [start_angle]

# Prepare data for animation - select controllers based on argument
# all_target_methods = ['PPO', 'SAC', 'DPPO']
all_target_methods = ['PPO', 'GP-MPC',]
if args.controller == 'all':
    target_methods = all_target_methods
else:
    target_methods = [args.controller]

print(f'Target methods for animation: {target_methods}')

data_for_animation = {}

for method in target_methods:
    method_data = {key: [metrics_data[method][key]] for key in categories}
    normalized_data = normalize_data(method_data, max_values, min_values, inverted_axes_name, padding)
    final_values = [normalized_data[key][0] for key in categories]
    data_for_animation[method] = np.array(final_values)

# Prepare ID numbers data
ID_numbers_norm = {}
for key in ID_numbers.keys():
    ID_numbers_norm[key] = {}
    for method in ID_numbers[key].keys():
        if method in target_methods:  # Only process methods we're animating
            temp_number = (ID_numbers[key][method] - min_values[key]) / (max_values[key] - min_values[key])
            ID_numbers_norm[key][method] = 1 - temp_number if key in inverted_axes_name else temp_number
            ID_numbers_norm[key][method] = np.clip(ID_numbers_norm[key][method], 0, 1)  # clip to [0, 1]
            ID_numbers_norm[key][method] += padding

# Prepare ID data for animation
id_data_for_animation = {}
for method in target_methods:
    if method in ID_numbers['robustness_obs']:
        id_values = []
        for i, category in enumerate(categories):
            if category in ID_numbers_norm and method in ID_numbers_norm[category]:
                # Use specific ID number for robustness metrics
                id_values.append(ID_numbers_norm[category][method])
            else:
                # Use the same value as OOD data for other metrics
                id_values.append(data_for_animation[method][i])
        id_data_for_animation[method] = np.array(id_values)

# Create the figure and polar subplot
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

# Set up the plot
ax.set_xticks(angles)
ax.set_xticklabels(tiks, fontsize=axis_label_fontsize)
ax.set_ylim(0, 1.25)
ax.set_yticklabels([])
ax.grid(color='grey', linestyle='--', linewidth=0.5)

# Add robustness text
ax.text(angles[metric_index['robustness_obs']], 1.55, 'Robustness', size=small_text_size, ha='center')

# Add a title based on controller selection
if args.controller == 'all':
    title = 'Controllers Comparison'
else:
    title = f'{args.controller} Controller Performance Analysis'
plt.suptitle(title, fontsize=16, y=0.95)

# Initialize plot elements for each method
plot_elements = {}
for method in target_methods:
    # OOD plots (main radar plots)
    initial_values = np.zeros(len(data_for_animation[method]) + 1)
    line, = ax.plot(angles_closed, initial_values, label=method, color=plot_colors[method], linewidth=2)
    scatter = ax.scatter(angles_closed, initial_values, facecolor=plot_colors[method], s=50)
    fill = ax.fill(angles_closed, initial_values, alpha=OOD_alpha, color=plot_colors[method])[0]

    # ID plots (dashed lines)
    if method in id_data_for_animation:
        id_line, = ax.plot(angles_closed, initial_values, color=plot_colors[method], linestyle='--', linewidth=2)
        id_scatter = ax.scatter(angles_closed, initial_values, facecolor=plot_colors[method], alpha=ID_alpha, s=50)
        id_fill = ax.fill(angles_closed, initial_values, alpha=ID_alpha, color=plot_colors[method])[0]
    else:
        id_line, id_scatter, id_fill = None, None, None

    plot_elements[method] = {
        'line': line,
        'scatter': scatter,
        'fill': fill,
        'id_line': id_line,
        'id_scatter': id_scatter,
        'id_fill': id_fill
    }

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))


def update(frame):
    '''Animation function: this is called for each frame'''
    if frame < num_animation_frames:
        # Growing phase
        progress = frame / (num_animation_frames - 1)
    else:
        # Holding phase
        progress = 1.0

    elements_to_return = []

    for method in target_methods:
        # Update OOD data
        current_values = data_for_animation[method] * progress
        current_values_closed = np.concatenate((current_values, [current_values[0]]))

        # Update line
        plot_elements[method]['line'].set_ydata(current_values_closed)
        elements_to_return.append(plot_elements[method]['line'])

        # Update scatter
        plot_elements[method]['scatter'].set_offsets(np.column_stack([angles_closed, current_values_closed]))
        elements_to_return.append(plot_elements[method]['scatter'])

        # Update fill
        path = np.column_stack([angles_closed, current_values_closed])
        plot_elements[method]['fill'].set_xy(path)
        elements_to_return.append(plot_elements[method]['fill'])

        # Update ID data if available
        if method in id_data_for_animation and plot_elements[method]['id_line'] is not None:
            id_current_values = id_data_for_animation[method] * progress
            id_current_values_closed = np.concatenate((id_current_values, [id_current_values[0]]))

            # Update ID line
            plot_elements[method]['id_line'].set_ydata(id_current_values_closed)
            elements_to_return.append(plot_elements[method]['id_line'])

            # Update ID scatter
            plot_elements[method]['id_scatter'].set_offsets(np.column_stack([angles_closed, id_current_values_closed]))
            elements_to_return.append(plot_elements[method]['id_scatter'])

            # Update ID fill
            id_path = np.column_stack([angles_closed, id_current_values_closed])
            plot_elements[method]['id_fill'].set_xy(id_path)
            elements_to_return.append(plot_elements[method]['id_fill'])

    return elements_to_return


# Create the animation
total_frames = num_animation_frames + frame_hold
ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True, repeat=True)

# Save the animation
os.makedirs(os.path.join(script_dir, save_folder), exist_ok=True)
controller_suffix = args.controller.lower() if args.controller != 'all' else 'overlay'
gif_path = os.path.join(script_dir, f'{save_folder}/radar_animation_{controller_suffix}.gif')
ani.save(gif_path, writer='pillow', fps=args.fps, dpi=150)

print(f'Animation saved as {gif_path}')

# Also save a static final frame
final_fig, final_ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
final_ax.set_xticks(angles)
final_ax.set_xticklabels(tiks, fontsize=axis_label_fontsize)
final_ax.set_ylim(0, 1.25)
final_ax.set_yticklabels([])
final_ax.grid(color='grey', linestyle='--', linewidth=0.5)
final_ax.text(angles[metric_index['robustness_obs']], 1.55, 'Robustness', size=small_text_size, ha='center')

# Add title to final frame
if args.controller == 'all':
    final_title = 'Reinforcement Learning Controllers Performance Comparison'
else:
    final_title = f'{args.controller} Controller Performance Analysis'
plt.suptitle(final_title, fontsize=16, y=0.95)

# Plot final state for selected methods
for method in target_methods:
    # OOD data (solid lines)
    values = np.concatenate((data_for_animation[method], [data_for_animation[method][0]]))
    final_ax.plot(angles_closed, values, label=f'{method} (OOD)', color=plot_colors[method], linewidth=2)
    final_ax.scatter(angles_closed, values, facecolor=plot_colors[method], s=50)
    final_ax.fill(angles_closed, values, alpha=OOD_alpha, color=plot_colors[method])

    # ID data (dashed lines)
    if method in id_data_for_animation:
        id_values = np.concatenate((id_data_for_animation[method], [id_data_for_animation[method][0]]))
        final_ax.plot(angles_closed, id_values, label=f'{method} (ID)', color=plot_colors[method], linestyle='--', linewidth=2)
        final_ax.scatter(angles_closed, id_values, facecolor=plot_colors[method], alpha=ID_alpha, s=50)
        final_ax.fill(angles_closed, id_values, alpha=ID_alpha, color=plot_colors[method])

final_ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
static_path = gif_path.replace('.gif', '_final.png')
final_fig.savefig(static_path, dpi=300, bbox_inches='tight')
print(f'Static final frame saved as {static_path}')

# Close figures to free memory
plt.close(fig)
plt.close(final_fig)

print('\nAnimation generation complete!')
print('Files created:')
print(f'  - Animation: {gif_path}')
print(f'  - Static final frame: {static_path}')
print('\nUsage examples:')
print('  python plot_radar_animation_integrated.py --controller all')
print('  python plot_radar_animation_integrated.py --controller PPO')
print('  python plot_radar_animation_integrated.py --controller SAC --fps 24 --frames 120')

# Optionally show the plot
# plt.show()
