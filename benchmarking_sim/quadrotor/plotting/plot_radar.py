import os
import sys
from pathlib import Path

# import seaborn
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn
from benchmarking_sim.quadrotor.benchmark_util.utils import plot_colors
from benchmarking_sim.quadrotor.benchmark_util.utils \
    import load_metric, plotting_data_dir, tag_ctrl_list

script_dir = os.path.dirname(__file__)
# set up Nature sytle plotting
# set up seaborn style
seaborn.set_palette('deep')
seaborn.set_context('paper', font_scale=1.5)

# set up matplotlib style
plt.rcParams.update({
    # 'font.family': 'arial',
    'grid.alpha': 1.0,
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
    # 'font.size': 20,
})

if len(sys.argv) > 2 and sys.argv[2] == 'abs':
    robustness_type = 'abs'
    save_folder = 'radar_abs'
else:
    robustness_type = 'relative'
    save_folder = 'radar'
    # robustness_type = 'abs'
    # save_folder = 'radar_abs'
print(f"Plotting with {robustness_type} robustness.")

# set up matplotlib parameters
transfer_metric = {}
axis_label_fontsize = 20
text_fontsize = 30
supertitle_fontsize = 30
subtitle_fontsize = 30
small_text_size = 20
padding = 0.15 # padding for the radar plot
OOD_alpha = 0.4 # alpha for the OOD data
ID_alpha = 0.15 

transfer_metric = load_metric(script_dir, transfer_metric, 'iLQR')
transfer_metric = load_metric(script_dir, transfer_metric, 'F-MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Nonlinear MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Linear MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'Geometric Control')
transfer_metric = load_metric(script_dir, transfer_metric, 'LQR')
transfer_metric = load_metric(script_dir, transfer_metric, 'GP-MPC')
transfer_metric = load_metric(script_dir, transfer_metric, 'PPO')
transfer_metric = load_metric(script_dir, transfer_metric, 'DPPO')
transfer_metric = load_metric(script_dir, transfer_metric, 'SAC')
transfer_metric = load_metric(script_dir, transfer_metric, 'PPO-MPC')

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
    'worst_generalization_performance': r'$\qquad\qquad\qquad\quad$  Generalization',
    'performance': r'$\qquad\qquad\qquad\quad$ Performance',
    'inference_time': 'Online     \nComputation     \n\n',
    'model_complexity': 'Required Model                    \nKnowledge                    ',
    'sampling_complexity': '\n\n\nSampling\ncomplexity',
    'robustness_proc': 'P',
    'robustness_obs': 'O',
    'robustness_param': r'$\theta$',
}

ID_numbers = {
    'robustness_obs': {
        'PPO': 60,
        'SAC': 200,
        'DPPO': 80,
        'PPO-MPC': 200,
    },
    'robustness_proc': {
        'PPO': 20,
        'SAC': 14,
        'DPPO': 25,
        'PPO-MPC': 8,
    },
    'robustness_param': {
        'PPO': 200,
        'SAC': 200,
        'DPPO': 200,
        'PPO-MPC': 13,
    },
    'worst_generalization_performance': {
        'PPO': 0.054727893216329365,
        'SAC': 0.02443952134274480,
        'DPPO': 0.05276080159023642,
        'PPO-MPC': 0.03715516806190815,
    }
}

ID_numbers_abs = {
    'robustness_obs': {
        'PPO': 100,
        'SAC': 0,
        'DPPO': 100,
        'PPO-MPC': 200,
    },
    'robustness_proc': {
        'PPO': 35,
        'SAC': 0,
        'DPPO': 30,
        'PPO-MPC': 8,
    },
    'robustness_param': {
        'PPO': 200,
        'SAC': 0.01,
        'DPPO': 200,
        'PPO-MPC': 13,
    },
    'worst_generalization_performance': {
        'PPO': 0.054727893216329365,
        'SAC': 0.02443952134274480,
        'DPPO': 0.05276080159023642,
        'PPO-MPC': 0.03715516806190815,
    }
}

if robustness_type == 'abs':
    ID_numbers = ID_numbers_abs

def load_robustness_failure_points():
    """Load robustness failure points from process_experiment_data.py results."""
    import json
    from pathlib import Path
    
    # Try to load the consolidated failure results
    script_dir = os.path.dirname(__file__)
    data_dir = plotting_data_dir(Path(script_dir))
    consolidated_file = data_dir / 'robustness_failure_points.json'
    
    if not consolidated_file.exists():
        print(f"Warning: Failure results file not found at {consolidated_file}")
        print("Using hardcoded robustness values. Run process_experiment_data.py all to refresh robustness data.")
        return {}
    
    try:
        with open(consolidated_file, 'r') as f:
            failure_data = json.load(f)
        
        print(f"Loaded failure results from: {consolidated_file}")
        return failure_data
    except Exception as e:
        print(f"Error loading failure results: {e}")
        print("Using hardcoded values.")
        return {}

def update_metrics_with_failure_data(metrics_data, failure_data, robustness_type):
    """Update metrics_data with failure points from process_experiment_data.py results."""
    
    if not failure_data:
        print("No failure data available, using existing hardcoded values.")
        return metrics_data
    
    # Choose the appropriate failure type
    if robustness_type == 'abs':
        failures = failure_data.get('absolute_failures', {})
        print("Using absolute failure thresholds (0.25m RMSE)")
    else:
        failures = failure_data.get('relative_failures', {})
        print("Using relative failure thresholds (200% degradation)")
    
    # Map noise types to metric names
    noise_to_metric = {
        'obs_noise': 'robustness_obs',
        'proc_noise': 'robustness_proc',  
        'param': 'robustness_param'
    }
    
    if robustness_type == 'abs':
        noise_to_metric = {
            'obs_noise': 'abs_robustness_obs',
            'proc_noise': 'abs_robustness_proc',
            'param': 'abs_robustness_param'
        }
    
    # Update metrics for each noise type
    for noise_type, metric_name in noise_to_metric.items():
        if noise_type in failures:
            print(f"\nUpdating {metric_name} from {noise_type} failure data:")
            
            for method, failure_point in failures[noise_type].items():
                if method in metrics_data:
                    old_value = metrics_data[method][metric_name]
                    metrics_data[method][metric_name] = failure_point
                    print(f"  {method:15}: {old_value} -> {failure_point}")
                else:
                    print(f"  Warning: Method '{method}' not found in metrics_data")
    
    return metrics_data

def update_id_numbers_with_failure_data(ID_numbers, failure_data, robustness_type):
    """Update the RL overlay robustness axes from DR robustness failures."""
    if not failure_data:
        return ID_numbers

    failures = failure_data.get(
        (
            'domain_randomization_absolute_failures'
            if robustness_type == 'abs'
            else 'domain_randomization_relative_failures'
        ),
        {}
    )
    noise_to_metric = {
        'obs_noise': 'robustness_obs',
        'proc_noise': 'robustness_proc',
        'param': 'robustness_param',
    }

    for noise_type, metric_name in noise_to_metric.items():
        if noise_type not in failures or metric_name not in ID_numbers:
            continue
        for method, failure_point in failures[noise_type].items():
            if method in ID_numbers[metric_name]:
                ID_numbers[metric_name][method] = failure_point

    return ID_numbers

def update_id_numbers_with_generalization_data(ID_numbers):
    """Update the RL overlay generalization axis from GEN generalization files."""
    if 'worst_generalization_performance' not in ID_numbers:
        return ID_numbers

    method_to_file = {
        'PPO': 'ppo',
        'SAC': 'sac',
        'DPPO': 'dppo',
        'PPO-MPC': 'ppo_mpc',
    }
    data_dir = plotting_data_dir(Path(script_dir))

    for method, file_stem in method_to_file.items():
        if method not in ID_numbers['worst_generalization_performance']:
            continue

        candidates = [
            data_dir / f'{file_stem}_gen_generalization.npy',
            data_dir / f'{file_stem}_domain_rand_generalization.npy',
        ]
        gen_file = next((path for path in candidates if path.exists()), None)
        if gen_file is None:
            if method in transfer_metric:
                ID_numbers['worst_generalization_performance'][method] = (
                    worst_generalization_rmse(method)
                )
            continue

        gen_data = np.load(gen_file, allow_pickle=True).item()
        rmse_values = []
        for episode_data in gen_data.get('generalization', {}).values():
            rmse = episode_data.get('rmse')
            if isinstance(rmse, dict) and 'mean' in rmse:
                rmse_values.append(rmse['mean'])
        if rmse_values:
            ID_numbers['worst_generalization_performance'][method] = (
                float(np.nanmax(rmse_values))
            )

    return ID_numbers

# Load failure data and update metrics
failure_data = load_robustness_failure_points()

def plot_id_number(ax, metric_name, model_name, angle, plot_colors, ID_numbers, ID_numbers_norm, small_text_size):
    """
    Plot an ID number point and label for a specific metric and model.
    
    Args:
        ax: Matplotlib axis object
        metric_name: Name of the metric (e.g., 'robustness_obs')
        model_name: Name of the model (e.g., 'PPO')
        angle: Angle position on the radar plot
        plot_colors: Dictionary mapping model names to colors
        ID_numbers: Dictionary with original ID numbers
        ID_numbers_norm: Dictionary with normalized ID numbers
        small_text_size: Font size for text labels
    """
    if model_name not in ID_numbers_norm[metric_name]:
        return
        
    # Get normalized value
    y_pos = ID_numbers_norm[metric_name][model_name]
    
    # Plot the point
    ax.scatter(
        angle, 
        y_pos, 
        facecolor=plot_colors[model_name], 
        # edgecolor='black', 
        # s=100, 
        # zorder=10
    )
    
    # Add text with original ID number value
    x_offset = -0.3 if metric_name == 'worst_generalization_performance' else 0
    
    ax.text(
        angle + x_offset, 
        y_pos,
        f'{ID_numbers[metric_name][model_name]}', 
        size=small_text_size
    )

def normalize_data(data, max_values, min_values, inverted_axes_name, lower_padding):
    """
    Normalize the data based on max and min values, applying padding and inversion where necessary.
    
    Args:
        data: Dictionary with raw data values
        max_values: Dictionary with maximum values for each metric
        min_values: Dictionary with minimum values for each metric
        inverted_axes_name: List of axes that should be inverted
        lower_padding: Padding to apply to the normalized values
    Returns:
        normalized_data: Dictionary with normalized values
    """
    normalized_data = {}
    
    for key in data.keys():
        temp_value = (np.array(data[key]) - min_values[key]) / (max_values[key] - min_values[key])
        temp_value = (1 - temp_value) if key in inverted_axes_name else temp_value
        temp_value = np.clip(temp_value, 0, 1)  # clip to [0, 1]
        normalized_data[key] = temp_value + lower_padding
    
    return normalized_data

def format_radar_value(value, metric_name=None):
    """Format numeric radar labels compactly."""
    if isinstance(value, np.ndarray):
        value = value.item()
    if not isinstance(value, (float, int, np.floating, np.integer)):
        return str(value)

    value = float(value)
    if metric_name in ['robustness_proc', 'robustness_obs', 'robustness_param']:
        return f'{value:.1f}'
    if metric_name in ['performance', 'worst_generalization_performance']:
        return f'{value:.2f}'
    return f'{value:.3f}'

def spider(df, 
           *, 
           id_column, 
           title=None, 
           subtitle=None, 
           max_values=None, 
           min_values=None,
           lower_padding=.25, 
           plt_name='',
           robustness_type='relative'):
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient='list') 
    ids = df[id_column].tolist() # ['controller1', 'controller2', ...]
    
    if max_values is None:
        max_values = {key: max(value) for key, value in data.items()}
    if min_values is None:
        min_values = {key: min(value) for key, value in data.items()}

    normalized_data = {
        key: 0 for key in data.keys()
    }
    
    # normalize the data
    normalized_data = normalize_data(data, max_values, min_values, 
                                     inverted_axes_name, lower_padding)

    # normalized ID numbers
    num_axis = len(data.keys()) # number of axes
    tiks = list(data.keys())
    tiks = [axis_legend_dict.get(tik, tik) for tik in tiks]  # replace keys with axis legend dict
    tiks += tiks[:1]
    start_angle = np.deg2rad(0)
    angles = [ 
        start_angle, # worst generalization
        np.deg2rad(60), # performance
        np.deg2rad(120), # inference time
        np.deg2rad(180), # model complexity
        np.deg2rad(240), # sampling complexity
        np.deg2rad(300 - 15), # robustness proc
        np.deg2rad(300), # observation noise
        np.deg2rad(300 + 15), # robustness param
        start_angle, # close the circle
    ]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True), )
    # ax.set_theta_offset(np.deg2rad(60))  # Rotate by 60 degrees
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]

        values += values[:1]  # Close the plot for a better look
        if model_name in ['MAX', 'MIN']:
            # ax.plot(angles, values, color=plot_colors[model_name], )
            # ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            # ax.fill(angles, values, alpha=1, color=plot_colors[model_name], )
            continue
        else:
            ax.plot(angles, values, 
                    label=model_name, 
                    color=plot_colors[model_name], )
            ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            ax.fill(angles, values, 
                    alpha=OOD_alpha, 
                    color=plot_colors[model_name], )

        if model_name in ['PPO', 'SAC', 'DPPO', 'PPO-MPC']:
            for metric_name in ID_numbers.keys():
                normalized_data[metric_name] = ID_numbers_norm[metric_name][model_name]
            values_ID = [normalized_data[key] for key in data.keys()]
            values_ID = [i.item() if isinstance(i, np.ndarray) else i for i in values_ID]
            values_ID += values_ID[:1]  # Close the plot for a better look
            ax.plot(angles, values_ID, label=model_name, color=plot_colors[model_name], linestyle='--')
            ax.scatter(angles, values_ID, facecolor=plot_colors[model_name], alpha=ID_alpha)
            ax.fill(angles, values_ID, alpha=ID_alpha, color=plot_colors[model_name])
            ax.text(
                angles[metric_index['robustness_proc']] - 0.25,
                values_ID[metric_index['robustness_proc']] + 0.1,
                format_radar_value(ID_numbers['robustness_proc'][model_name], 'robustness_proc'),
                size=small_text_size,
                color='black',
            )
            ax.text(
                angles[metric_index['robustness_obs']] - 0.15,
                values_ID[metric_index['robustness_obs']] + 0.1,
                format_radar_value(ID_numbers['robustness_obs'][model_name], 'robustness_obs'),
                size=small_text_size,
                color='black',
            )
            ax.text(
                angles[metric_index['robustness_param']] + 0.0,
                values_ID[metric_index['robustness_param']] + 0.,
                format_radar_value(ID_numbers['robustness_param'][model_name], 'robustness_param'),
                size=small_text_size,
                color='black',
            )
            # Format worst_generalization_performance with 2 decimal places
            worst_gen_value = ID_numbers['worst_generalization_performance'][model_name]
            formatted_worst_gen = f'{worst_gen_value:.2f}' if isinstance(worst_gen_value, float) else str(worst_gen_value)
            ax.text(
                angles[metric_index['worst_generalization_performance']] - 0.15, 
                values_ID[metric_index['worst_generalization_performance']], 
                formatted_worst_gen, 
                size=small_text_size,
                color='black',
            )

        # customize the text
        for _x, _y, t in zip(angles, values, actual_values):
            axis_midpoint = lower_padding + 0.5
            metric_name = None
            for name, idx in metric_index.items():
                if _x == angles[idx]:
                    metric_name = name
                    break

            if _x == angles[metric_index['inference_time']]:
                t = f'{t:.1E}' if isinstance(t, float) else str(t)
            elif _x == angles[metric_index['sampling_complexity']]:
                t = '0' if t == int(1) else f'{t:.1E}' # write number in scientific notation
            elif _x == angles[metric_index['model_complexity']]:
                if t == 3: t = 'Model-free'
                if t == 2: t = '   Linear\n   model'
                if t == 1: t = '   Partially uncertain \n nonlinear model'
                if model_name == 'PID': t = '   Kinematic \n   model'
                if t == 0: t = 'Perfect nonlinear\n   model'
            elif _x in [
                angles[metric_index['robustness_proc']],
                angles[metric_index['robustness_obs']],
                angles[metric_index['robustness_param']],
            ]:
                t = format_radar_value(t, metric_name)
            elif _x in [angles[metric_index['performance']], angles[metric_index['worst_generalization_performance']]]:
                t = f'{t:.2f}' if isinstance(t, float) else str(t)  # 2 decimal places for performance and generalization
            else:
                t = f'{t:.3f}' if isinstance(t, float) else str(t)

            if _x == angles[metric_index['worst_generalization_performance']]:
                ax.text(_x + 0.05, _y - 0.1, t, size=small_text_size)
            elif _x == angles[metric_index['performance']]:
                ax.text(
                    _x,
                    axis_midpoint + 0.2*(_y-0.5),
                    t,
                    size=small_text_size,
                    ha='center',
                    va='center',
                )
            elif _x == angles[metric_index['inference_time']]:
                ax.text(
                    _x,
                    axis_midpoint + 0.2*(_y-0.5),
                    t,
                    size=small_text_size,
                    ha='center',
                    va='center',
                )
            elif _x == angles[metric_index['model_complexity']]:
                ax.text(
                    _x,
                    axis_midpoint,
                    t,
                    size=small_text_size,
                    ha='center',
                    va='center',
                )
            elif _x == angles[metric_index['sampling_complexity']]:
                ax.text(
                    _x,
                    axis_midpoint + 0.2*(_y-0.5),
                    t,
                    size=small_text_size,
                    ha='center',
                    va='center',
                )
            elif _x == angles[metric_index['robustness_proc']]:
                ax.text(_x - 0.15, _y, t, size=small_text_size)
            elif _x == angles[metric_index['robustness_obs']]:
                ax.text(_x - 0.1, _y, t, size=small_text_size)
            elif _x == angles[metric_index['robustness_param']]:
                ax.text(_x + 0.0, _y, t, size=small_text_size)
            else: # shift all the other axes 
                ax.text(_x, _y - 0.01, t, size=small_text_size)
            
            # # plot extra dots for ID numbers
            # for metric_name in ['robustness_obs', 'robustness_proc', 'robustness_param', 'worst_generalization_performance']:
            #     if _x == angles[metric_index[metric_name]]:
            #         plot_id_number(
            #             ax=ax,
            #             metric_name=metric_name,
            #             model_name=model_name,
            #             angle=_x,
            #             plot_colors=plot_colors,
            #             ID_numbers=ID_numbers,
            #             ID_numbers_norm=ID_numbers_norm,
            #             small_text_size=small_text_size
            #         )
        
            
    # add additional text for robustness axes
    ax.text(angles[metric_index['robustness_obs']], 1.55, 'Robustness', size=small_text_size)
    # ax.fill(angles, 1.5*np.ones(num_axis + 1), alpha=0.7, color='lightgray')
    ax.set_ylim(0, 1.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks, fontsize=axis_label_fontsize)
    if title is not None: plt.suptitle(title, fontsize=supertitle_fontsize)
    if subtitle is not None: plt.title(subtitle, fontsize=subtitle_fontsize)
    os.makedirs(os.path.join(script_dir, save_folder), exist_ok=True)
    fig_save_path = os.path.join(script_dir, f'{save_folder}/radar_{plt_name}.pdf')
    fig.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    fig.savefig(fig_save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'figure saved as {fig_save_path}')

radar = spider
methods = [
    'GP-MPC', 'Linear MPC', 'Nonlinear MPC', 'F-MPC', 
    'PPO', 'SAC', 'DPPO', 'PPO-MPC', 
    'Geometric Control', 'iLQR', 'LQR'
]

def worst_generalization_rmse(method):
    """Return the worst RMSE across all processed generalization episode lengths."""
    return float(np.nanmax(transfer_metric[method]['rmse']))

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
        'abs_robustness_proc': 0,
        'abs_robustness_obs': 0,
        'abs_robustness_param': 0,
    }

######################## Nominal ###############################
# Fill in the data
# GP-MPC
# metrics_data['GP-MPC']['worst_generalization_performance'] = 0.03286539605493106 # worst_generalization_rmse('GP-MPC')
# metrics_data['GP-MPC']['performance'] = 0.0147  # Manually set
metrics_data['GP-MPC']['worst_generalization_performance'] = worst_generalization_rmse('GP-MPC')
metrics_data['GP-MPC']['performance'] = transfer_metric['GP-MPC']['rmse'][2]
metrics_data['GP-MPC']['inference_time'] = transfer_metric['GP-MPC']['inference_time']
metrics_data['GP-MPC']['model_complexity'] = 1
metrics_data['GP-MPC']['sampling_complexity'] = 660
metrics_data['GP-MPC']['robustness_proc'] = 5
metrics_data['GP-MPC']['robustness_obs'] = 50
metrics_data['GP-MPC']['robustness_param'] = 4.5
metrics_data['GP-MPC']['abs_robustness_obs'] = 100
metrics_data['GP-MPC']['abs_robustness_proc'] = 8
metrics_data['GP-MPC']['abs_robustness_param'] = 5.2

# Linear MPC
metrics_data['Linear MPC']['worst_generalization_performance'] = worst_generalization_rmse('Linear MPC')
# metrics_data['Linear MPC']['worst_generalization_performance'] = 0.036
metrics_data['Linear MPC']['performance'] = transfer_metric['Linear MPC']['rmse'][2]
# metrics_data['Linear MPC']['performance'] = 0.029
metrics_data['Linear MPC']['inference_time'] = transfer_metric['Linear MPC']['inference_time']
metrics_data['Linear MPC']['model_complexity'] = 2
metrics_data['Linear MPC']['sampling_complexity'] = 1
metrics_data['Linear MPC']['robustness_proc'] = 6
metrics_data['Linear MPC']['robustness_obs'] = 100
metrics_data['Linear MPC']['robustness_param'] = 4.8
metrics_data['Linear MPC']['abs_robustness_obs'] = 100
metrics_data['Linear MPC']['abs_robustness_proc'] = 7
metrics_data['Linear MPC']['abs_robustness_param'] = 5.2

# Nonlinear MPC
metrics_data['Nonlinear MPC']['worst_generalization_performance'] = worst_generalization_rmse('Nonlinear MPC')
# metrics_data['Nonlinear MPC']['worst_generalization_performance'] = 0.024
metrics_data['Nonlinear MPC']['performance'] = transfer_metric['Nonlinear MPC']['rmse'][2]
# metrics_data['Nonlinear MPC']['performance'] = 0.008
metrics_data['Nonlinear MPC']['inference_time'] = transfer_metric['Nonlinear MPC']['inference_time']
# metrics_data['Nonlinear MPC']['inference_time'] = 5.5e-4
metrics_data['Nonlinear MPC']['model_complexity'] = 0
metrics_data['Nonlinear MPC']['sampling_complexity'] = 1
metrics_data['Nonlinear MPC']['robustness_proc'] = 3
metrics_data['Nonlinear MPC']['robustness_obs'] = 50
metrics_data['Nonlinear MPC']['robustness_param'] = 1.4
metrics_data['Nonlinear MPC']['abs_robustness_obs'] = 100
metrics_data['Nonlinear MPC']['abs_robustness_proc'] = 8
metrics_data['Nonlinear MPC']['abs_robustness_param'] = 5.0

# F-MPC
metrics_data['F-MPC']['worst_generalization_performance'] = worst_generalization_rmse('F-MPC')
metrics_data['F-MPC']['performance'] = transfer_metric['F-MPC']['rmse'][2]
metrics_data['F-MPC']['inference_time'] = transfer_metric['F-MPC']['inference_time']
metrics_data['F-MPC']['model_complexity'] = 0
metrics_data['F-MPC']['sampling_complexity'] = 1
metrics_data['F-MPC']['robustness_proc'] = 2
metrics_data['F-MPC']['robustness_obs'] = 30
metrics_data['F-MPC']['robustness_param'] = 1.6
metrics_data['F-MPC']['abs_robustness_obs'] = 90
metrics_data['F-MPC']['abs_robustness_proc'] = 7
metrics_data['F-MPC']['abs_robustness_param'] = 5.0

# PPO
metrics_data['PPO']['worst_generalization_performance'] = worst_generalization_rmse('PPO')
metrics_data['PPO']['performance'] = transfer_metric['PPO']['rmse'][2]
metrics_data['PPO']['inference_time'] = 1.4064e-04  #1.4255e-04
metrics_data['PPO']['model_complexity'] = 3
metrics_data['PPO']['sampling_complexity'] = 2019600 #739200
metrics_data['PPO']['robustness_obs'] = 14
metrics_data['PPO']['robustness_proc'] = 4
metrics_data['PPO']['robustness_param'] = 1.5
metrics_data['PPO']['abs_robustness_obs'] = 60
metrics_data['PPO']['abs_robustness_proc'] = 16
metrics_data['PPO']['abs_robustness_param'] = 6.0

# SAC
metrics_data['SAC']['worst_generalization_performance'] = worst_generalization_rmse('SAC')
metrics_data['SAC']['performance'] = transfer_metric['SAC']['rmse'][2]
metrics_data['SAC']['inference_time'] = 9.27636118e-05  #9.10130414e-05
metrics_data['SAC']['model_complexity'] = 3
metrics_data['SAC']['sampling_complexity'] = 607200 #250800
metrics_data['SAC']['robustness_obs'] = 45
metrics_data['SAC']['robustness_proc'] = 4
metrics_data['SAC']['robustness_param'] = 2.5
metrics_data['SAC']['abs_robustness_obs'] = 200
metrics_data['SAC']['abs_robustness_proc'] = 7
metrics_data['SAC']['abs_robustness_param'] = 5.0

# DPPO
metrics_data['DPPO']['worst_generalization_performance'] = worst_generalization_rmse('DPPO')
metrics_data['DPPO']['performance'] = transfer_metric['DPPO']['rmse'][2]
metrics_data['DPPO']['inference_time'] = 1.0555e-04 #0.00014255
metrics_data['DPPO']['model_complexity'] = 3
metrics_data['DPPO']['sampling_complexity'] = 2310000 #712800
metrics_data['DPPO']['robustness_obs'] = 12
metrics_data['DPPO']['robustness_proc'] = 4
metrics_data['DPPO']['robustness_param'] = 2.0
metrics_data['DPPO']['abs_robustness_obs'] = 40
metrics_data['DPPO']['abs_robustness_proc'] = 14
metrics_data['DPPO']['abs_robustness_param'] = 4.0

# PPO-MPC
metrics_data['PPO-MPC']['worst_generalization_performance'] = worst_generalization_rmse('PPO-MPC')
metrics_data['PPO-MPC']['performance'] = transfer_metric['PPO-MPC']['rmse'][2]
metrics_data['PPO-MPC']['inference_time'] = 1.80392e-03 #5.5e-4
metrics_data['PPO-MPC']['model_complexity'] = 1
metrics_data['PPO-MPC']['sampling_complexity'] = 290400 #224400
metrics_data['PPO-MPC']['robustness_obs'] = 35
metrics_data['PPO-MPC']['robustness_proc'] = 3
metrics_data['PPO-MPC']['robustness_param'] = 2.0
metrics_data['PPO-MPC']['abs_robustness_obs'] = 200
metrics_data['PPO-MPC']['abs_robustness_proc'] = 9
metrics_data['PPO-MPC']['abs_robustness_param'] = 7.

# PID
metrics_data['Geometric Control']['worst_generalization_performance'] = worst_generalization_rmse('Geometric Control')
metrics_data['Geometric Control']['performance'] = transfer_metric['Geometric Control']['rmse'][2]
metrics_data['Geometric Control']['inference_time'] = transfer_metric['Geometric Control']['inference_time']
metrics_data['Geometric Control']['model_complexity'] = 1
metrics_data['Geometric Control']['sampling_complexity'] = 1
metrics_data['Geometric Control']['robustness_proc'] = 12
metrics_data['Geometric Control']['robustness_obs'] = 100
metrics_data['Geometric Control']['robustness_param'] = 4.5
metrics_data['Geometric Control']['abs_robustness_obs'] = 100
metrics_data['Geometric Control']['abs_robustness_proc'] = 14
metrics_data['Geometric Control']['abs_robustness_param'] = 4.5

# iLQR
metrics_data['iLQR']['worst_generalization_performance'] = worst_generalization_rmse('iLQR')
metrics_data['iLQR']['performance'] = transfer_metric['iLQR']['rmse'][2]
metrics_data['iLQR']['inference_time'] = transfer_metric['iLQR']['inference_time']
metrics_data['iLQR']['model_complexity'] = 0
metrics_data['iLQR']['sampling_complexity'] = 1
metrics_data['iLQR']['robustness_proc'] = 3
metrics_data['iLQR']['robustness_obs'] = 16
metrics_data['iLQR']['robustness_param'] = 0.6
metrics_data['iLQR']['abs_robustness_obs'] = 100
metrics_data['iLQR']['abs_robustness_proc'] = 12
metrics_data['iLQR']['abs_robustness_param'] = 5.0

# LQR
metrics_data['LQR']['worst_generalization_performance'] = worst_generalization_rmse('LQR')
metrics_data['LQR']['performance'] = transfer_metric['LQR']['rmse'][2]
metrics_data['LQR']['inference_time'] = transfer_metric['LQR']['inference_time']
metrics_data['LQR']['model_complexity'] = 2
metrics_data['LQR']['sampling_complexity'] = 1
metrics_data['LQR']['robustness_proc'] = 10
metrics_data['LQR']['robustness_obs'] = 100
metrics_data['LQR']['robustness_param'] = 5.0
metrics_data['LQR']['abs_robustness_obs'] = 60
metrics_data['LQR']['abs_robustness_proc'] = 14
metrics_data['LQR']['abs_robustness_param'] = 5.0

# Update metrics with failure data from process_experiment_data.py
metrics_data = update_metrics_with_failure_data(metrics_data, failure_data, robustness_type)
ID_numbers = update_id_numbers_with_failure_data(ID_numbers, failure_data, robustness_type)
ID_numbers = update_id_numbers_with_generalization_data(ID_numbers)

max_values = {key: max([metrics_data[method][key] for method in metrics_data.keys()]) for key in metrics_data['GP-MPC'].keys()}
min_values = {key: min([metrics_data[method][key] for method in metrics_data.keys()]) for key in metrics_data['GP-MPC'].keys()}

for metric_name, method_values in ID_numbers.items():
    if metric_name not in max_values:
        continue
    id_values = list(method_values.values())
    if id_values:
        max_values[metric_name] = max(max_values[metric_name], max(id_values))
        min_values[metric_name] = min(min_values[metric_name], min(id_values))

# handtune max and min to make the plot look better
if robustness_type == 'abs':
    for r_type in ['proc', 'obs', 'param']:
        abs_key = f'abs_robustness_{r_type}'
        rel_key = f'robustness_{r_type}'
        max_values[rel_key] = max([metrics_data[method][abs_key] for method in metrics_data.keys()])
        min_values[rel_key] = min([metrics_data[method][abs_key] for method in metrics_data.keys()])
max_values['performance'] = 0.04
max_values['worst_generalization_performance'] = 0.087
max_values['sampling_complexity'] = 400000
max_values['robustness_proc'] = 10
max_values['robustness_obs'] = 100
max_values['robustness_param'] = 5.0
max_values['abs_robustness_obs'] = 100
max_values['abs_robustness_proc'] = 10.
max_values['abs_robustness_param'] = 5.0
max_values['inference_time'] = 1.0e-2

for metric_name, method_values in ID_numbers.items():
    if metric_name not in max_values:
        continue
    id_values = list(method_values.values())
    if id_values:
        max_values[metric_name] = max(max_values[metric_name], max(id_values))
        min_values[metric_name] = min(min_values[metric_name], min(id_values))


shared_performance_axis = False
shared_performance_axis = True
if shared_performance_axis:
    # merge take the crosssection of max and min performance and worst generalization performance
    performance_max = max(max_values['performance'], max_values['worst_generalization_performance'])
    performance_min = min(min_values['performance'], min_values['worst_generalization_performance'])
    max_values['performance'] = performance_max
    max_values['worst_generalization_performance'] = performance_max
    min_values['performance'] = performance_min
    min_values['worst_generalization_performance'] = performance_min 

# prepare the ID data
ID_numbers_norm = {
    metric: {model: 0.0 for model in models} 
    for metric, models in ID_numbers.items()
}
# normalize the ID numbeer acording to the max and min values of each category
for key in ID_numbers.keys():
    for method in ID_numbers[key].keys():
        temp_number = (ID_numbers[key][method] - min_values[key]) / (max_values[key] - min_values[key])
        ID_numbers_norm [key][method] = 1 - temp_number if key in inverted_axes_name else temp_number
        ID_numbers_norm [key][method] = np.clip(ID_numbers_norm[key][method], 0, 1)  # clip to [0, 1]
        ID_numbers_norm [key][method] += padding

# append the max and min values to the data
# read the argv
if len(sys.argv) > 1:
    algo = sys.argv[1]
else:
    # algo = 'pid'
    algo = 'ppo'

# convert the algo to the correct name in tag_ctrl_list
if algo in tag_ctrl_list.values():
    algo = list(tag_ctrl_list.keys())[list(tag_ctrl_list.values()).index(algo)]
assert algo in tag_ctrl_list.keys(), f'Algorithm {algo} not found in tag_ctrl_list. Available algorithms: {list(tag_ctrl_list.keys())}'

# initialize the data for the radar plot
data = {key: 0 for key in metrics_data['GP-MPC'].keys()}
for key in metrics_data['GP-MPC'].keys():
    data[key] = [metrics_data[algo][key]]
algos = [algo]

if robustness_type == 'abs':
    robustness_proc_val = data['abs_robustness_proc']
    robustness_obs_val = data['abs_robustness_obs']
    robustness_param_val = data['abs_robustness_param']
else:
    robustness_proc_val = data['robustness_proc']
    robustness_obs_val = data['robustness_obs']
    robustness_param_val = data['robustness_param']

spider(
    pd.DataFrame({
        'x': algos,
        'worst_generalization_performance': data['worst_generalization_performance'],
        'performance': data['performance'],
        'inference_time': data['inference_time'],
        'model_complexity': data['model_complexity'],
        'sampling_complexity': data['sampling_complexity'],
        'robustness_proc': robustness_proc_val,
        'robustness_obs': robustness_obs_val,
        'robustness_param': robustness_param_val,
    }),
    id_column='x',
    title=None,
    lower_padding=padding,
    plt_name=tag_ctrl_list[algo],
    max_values=max_values,
    min_values=min_values,
    robustness_type=robustness_type,
)
