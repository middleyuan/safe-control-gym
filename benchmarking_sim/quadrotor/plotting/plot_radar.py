import os
import sys

# import seaborn
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn
from benchmarking_sim.quadrotor.benchmark_util.utils \
    import plot_colors, load_metric, tag_ctrl_list

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
    # 'savefig.transparent': True,
    # 'font.size': 20,
})

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
transfer_metric = load_metric(script_dir, transfer_metric, 'PID')
transfer_metric = load_metric(script_dir, transfer_metric, 'LQR')
transfer_metric = load_metric(script_dir, transfer_metric, 'GP-MPC')

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
    'worst_generalization_performance': '$\qquad\qquad\qquad\quad$  Generalization',
    'performance': '$\qquad\qquad\qquad\quad$ Performance',
    'inference_time': 'Online     \nComputation     \n\n',
    'model_complexity': 'Required Model                    \nKnowledge                    ',
    'sampling_complexity': '\n\n\nSampling\ncomplexity',
    'robustness_proc': 'P',
    'robustness_obs': 'O',
    'robustness_param': r'$\theta$',
}

ID_numbers = {
    'robustness_proc': {
        'PPO': 20,
        'SAC': 12,
        'DPPO': 35,
    },
    'robustness_obs': {
        'PPO': 80,
        'SAC': 100,
        'DPPO': 90,
    },
    'robustness_param': {
        'PPO': 4,
        'SAC': 5,
        'DPPO': 5,
    },
    'worst_generalization_performance': {
        'PPO': 0.043,
        'SAC': 0.084,
        'DPPO': 0.07
    }
}

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

def spider(df, 
           *, 
           id_column, 
           title=None, 
           subtitle=None, 
           max_values=None, 
           min_values=None,
           lower_padding=.25, 
           plt_name=''):
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

        if model_name in ['PPO', 'SAC', 'DPPO']:
            # create a data list but replace the values with ID numbers
            # values_ID = 
            for metric_name in ID_numbers.keys():
                normalized_data[metric_name] = ID_numbers_norm[metric_name][model_name]
            values_ID = [normalized_data[key] for key in data.keys()]
            values_ID = [i.item() if isinstance(i, np.ndarray) else i for i in values_ID]
            values_ID += values_ID[:1]  # Close the plot for a better look
            # plot the ID numbers
            ax.plot(angles, values_ID, label=model_name, color=plot_colors[model_name], 
                    linestyle='--', )
            ax.scatter(angles, values_ID, facecolor=plot_colors[model_name], 
                       alpha=ID_alpha,)
            ax.fill(angles, values_ID, 
                    alpha=ID_alpha, 
                    color=plot_colors[model_name], )
            ax.text(angles[metric_index['robustness_proc']], 
                    values_ID[metric_index['robustness_proc']], 
                    ID_numbers['robustness_proc'][model_name], size=small_text_size)
            ax.text(angles[metric_index['robustness_obs']], 
                    values_ID[metric_index['robustness_obs']], 
                    ID_numbers['robustness_obs'][model_name], size=small_text_size)
            ax.text(angles[metric_index['robustness_param']], 
                    values_ID[metric_index['robustness_param']], 
                    ID_numbers['robustness_param'][model_name], size=small_text_size)
            ax.text(angles[metric_index['worst_generalization_performance']], 
                    values_ID[metric_index['worst_generalization_performance']], 
                    ID_numbers['worst_generalization_performance'][model_name], size=small_text_size)
                
       
        # customize the text
        for _x, _y, t in zip(angles, values, actual_values):
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
            elif _x == angles[metric_index['robustness_param']]:
                t = f'{t:.1f}' if isinstance(t, float) else str(t)
            else:
                t = f'{t:.3f}' if isinstance(t, float) else str(t)

            if _x == angles[metric_index['worst_generalization_performance']]:
                ax.text(_x - 0.15, _y - 0.1, t, size=small_text_size)
            elif _x == angles[metric_index['model_complexity']]:
                ax.text(_x + 0.3, _y + 0.2, t, size=small_text_size)
            elif _x == angles[metric_index['sampling_complexity']]:
                ax.text(_x - 0.15, _y + 0.2, t, size=small_text_size)
            elif _x in [angles[metric_index['robustness_proc']], 
                        angles[metric_index['robustness_obs']],
                        angles[metric_index['robustness_param']]]:
                ax.text(_x - 0., _y, t, size=small_text_size)
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
    os.makedirs(os.path.join(script_dir, 'radar'), exist_ok=True)
    fig_save_path = os.path.join(script_dir, f'radar/radar_{plt_name}.pdf')
    fig.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    fig.savefig(fig_save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'figure saved as {fig_save_path}')

radar = spider
methods = [
    'GP-MPC', 'Linear MPC', 'Nonlinear MPC', 'F-MPC', 
    'PPO', 'SAC', 'DPPO', 'PPO-MPC', 
    'PID', 'iLQR', 'LQR'
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

# Fill in the data
# GP-MPC
# metrics_data['GP-MPC']['worst_generalization_performance'] = 0.03286539605493106 # max(transfer_metric['GP-MPC']['rmse'][0], transfer_metric['GP-MPC']['rmse'][-1])
# metrics_data['GP-MPC']['performance'] = 0.0147  # Manually set
metrics_data['GP-MPC']['worst_generalization_performance'] = max(transfer_metric['GP-MPC']['rmse'][0], transfer_metric['GP-MPC']['rmse'][-1])
metrics_data['GP-MPC']['performance'] = transfer_metric['GP-MPC']['rmse'][2]
metrics_data['GP-MPC']['inference_time'] = transfer_metric['GP-MPC']['inference_time']
metrics_data['GP-MPC']['model_complexity'] = 1
metrics_data['GP-MPC']['sampling_complexity'] = 660
metrics_data['GP-MPC']['robustness_proc'] = 5
metrics_data['GP-MPC']['robustness_obs'] = 50
metrics_data['GP-MPC']['robustness_param'] = 4.5


# Linear MPC
metrics_data['Linear MPC']['worst_generalization_performance'] = max(transfer_metric['Linear MPC']['rmse'][0], transfer_metric['Linear MPC']['rmse'][-1])
# metrics_data['Linear MPC']['worst_generalization_performance'] = 0.036
metrics_data['Linear MPC']['performance'] = transfer_metric['Linear MPC']['rmse'][2]
# metrics_data['Linear MPC']['performance'] = 0.029
metrics_data['Linear MPC']['inference_time'] = transfer_metric['Linear MPC']['inference_time']
metrics_data['Linear MPC']['model_complexity'] = 2
metrics_data['Linear MPC']['sampling_complexity'] = 1
metrics_data['Linear MPC']['robustness_proc'] = 6
metrics_data['Linear MPC']['robustness_obs'] = 120
metrics_data['Linear MPC']['robustness_param'] = 4.8

# Nonlinear MPC
metrics_data['Nonlinear MPC']['worst_generalization_performance'] = max(transfer_metric['Nonlinear MPC']['rmse'][0], transfer_metric['Nonlinear MPC']['rmse'][-1])
# metrics_data['Nonlinear MPC']['worst_generalization_performance'] = 0.024
metrics_data['Nonlinear MPC']['performance'] = transfer_metric['Nonlinear MPC']['rmse'][2]
# metrics_data['Nonlinear MPC']['performance'] = 0.008
# metrics_data['Nonlinear MPC']['inference_time'] = transfer_metric['Nonlinear MPC']['inference_time']
metrics_data['Nonlinear MPC']['inference_time'] = 5.5e-4
metrics_data['Nonlinear MPC']['model_complexity'] = 0
metrics_data['Nonlinear MPC']['sampling_complexity'] = 1
metrics_data['Nonlinear MPC']['robustness_proc'] = 3
metrics_data['Nonlinear MPC']['robustness_obs'] = 50
metrics_data['Nonlinear MPC']['robustness_param'] = 1.4

# F-MPC
metrics_data['F-MPC']['worst_generalization_performance'] = max(transfer_metric['F-MPC']['rmse'][0], transfer_metric['F-MPC']['rmse'][-1])
metrics_data['F-MPC']['performance'] = transfer_metric['F-MPC']['rmse'][2]
metrics_data['F-MPC']['inference_time'] = transfer_metric['F-MPC']['inference_time']
metrics_data['F-MPC']['model_complexity'] = 0
metrics_data['F-MPC']['sampling_complexity'] = 1
metrics_data['F-MPC']['robustness_proc'] = 2
metrics_data['F-MPC']['robustness_obs'] = 30
metrics_data['F-MPC']['robustness_param'] = 1.6

# PPO
metrics_data['PPO']['worst_generalization_performance'] = max(0.11419083, 0.107995445)
metrics_data['PPO']['performance'] = 0.012809196573431799
metrics_data['PPO']['inference_time'] = 7.31e-5
metrics_data['PPO']['model_complexity'] = 3
metrics_data['PPO']['sampling_complexity'] = int(1e6)
metrics_data['PPO']['robustness_proc'] = 4
metrics_data['PPO']['robustness_obs'] = 18
metrics_data['PPO']['robustness_param'] = 1.2

# SAC
metrics_data['SAC']['worst_generalization_performance'] = max(0.20906559, 0.08379055)
metrics_data['SAC']['performance'] = 0.032350691213897304
metrics_data['SAC']['inference_time'] = 8.72e-5
metrics_data['SAC']['model_complexity'] = 3
metrics_data['SAC']['sampling_complexity'] = int(0.8e6)
metrics_data['SAC']['robustness_proc'] = 6
metrics_data['SAC']['robustness_obs'] = 100
metrics_data['SAC']['robustness_param'] = 2.0

# DPPO
metrics_data['DPPO']['worst_generalization_performance'] = max(0.14269699, 0.13267572)
metrics_data['DPPO']['performance'] = 0.021298943332121172
metrics_data['DPPO']['inference_time'] = 7.28e-5
metrics_data['DPPO']['model_complexity'] = 3
metrics_data['DPPO']['sampling_complexity'] = int(1e6)
metrics_data['DPPO']['robustness_proc'] = 7
metrics_data['DPPO']['robustness_obs'] = 25
metrics_data['DPPO']['robustness_param'] = 1.8

# PPO-MPC
metrics_data['PPO-MPC']['worst_generalization_performance'] = max(0.04358512, 0.01077949)
metrics_data['PPO-MPC']['performance'] = 0.01362814727788425
metrics_data['PPO-MPC']['inference_time'] = 5.5e-4
metrics_data['PPO-MPC']['model_complexity'] = 1
metrics_data['PPO-MPC']['sampling_complexity'] = int(0.4e6)
metrics_data['PPO-MPC']['robustness_proc'] = 3
metrics_data['PPO-MPC']['robustness_obs'] = 45
metrics_data['PPO-MPC']['robustness_param'] = 2.8

# PID
metrics_data['PID']['worst_generalization_performance'] = max(transfer_metric['PID']['rmse'][0], transfer_metric['PID']['rmse'][-1])
metrics_data['PID']['performance'] = transfer_metric['PID']['rmse'][2]
metrics_data['PID']['inference_time'] = transfer_metric['PID']['inference_time']
metrics_data['PID']['model_complexity'] = 1
metrics_data['PID']['sampling_complexity'] = 1
metrics_data['PID']['robustness_proc'] = 12
metrics_data['PID']['robustness_obs'] = 120
metrics_data['PID']['robustness_param'] = 4.5

# iLQR
metrics_data['iLQR']['worst_generalization_performance'] = max(transfer_metric['iLQR']['rmse'][0], transfer_metric['iLQR']['rmse'][-1])
metrics_data['iLQR']['performance'] = transfer_metric['iLQR']['rmse'][2]
metrics_data['iLQR']['inference_time'] = transfer_metric['iLQR']['inference_time']
metrics_data['iLQR']['model_complexity'] = 0
metrics_data['iLQR']['sampling_complexity'] = 1
metrics_data['iLQR']['robustness_proc'] = 3
metrics_data['iLQR']['robustness_obs'] = 16
metrics_data['iLQR']['robustness_param'] = 0.6

# LQR
metrics_data['LQR']['worst_generalization_performance'] = max(transfer_metric['LQR']['rmse'][0], transfer_metric['LQR']['rmse'][-1])
metrics_data['LQR']['performance'] = transfer_metric['LQR']['rmse'][2]
metrics_data['LQR']['inference_time'] = transfer_metric['LQR']['inference_time']
metrics_data['LQR']['model_complexity'] = 2
metrics_data['LQR']['sampling_complexity'] = 1
metrics_data['LQR']['robustness_proc'] = 10
metrics_data['LQR']['robustness_obs'] = 120
metrics_data['LQR']['robustness_param'] = 5.0

max_values = {key: max([metrics_data[method][key] for method in metrics_data.keys()]) for key in metrics_data['GP-MPC'].keys()}
min_values = {key: min([metrics_data[method][key] for method in metrics_data.keys()]) for key in metrics_data['GP-MPC'].keys()}

# handtune max and min to make the plot look better
max_values['performance'] = 0.04
max_values['worst_generalization_performance'] = 0.07
max_values['robustness_proc'] = 10
max_values['inference_time'] = 1.7e-3

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

spider(
    pd.DataFrame({
        'x': algos,
        'worst_generalization_performance': data['worst_generalization_performance'],
        'performance': data['performance'],
        'inference_time': data['inference_time'],
        'model_complexity': data['model_complexity'],
        'sampling_complexity': data['sampling_complexity'],
        'robustness_proc': data['robustness_proc'],
        'robustness_obs': data['robustness_obs'],
        'robustness_param': data['robustness_param'],
    }),
    id_column='x',
    title=None,
    lower_padding=padding,
    plt_name=algo,
    max_values=max_values,
    min_values=min_values,
)
