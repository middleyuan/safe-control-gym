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
# seaborn.set(style='whitegrid', palette='deep')
# seaborn.set_style('whitegrid')
seaborn.set_palette('deep')
seaborn.set_context('paper', font_scale=1.5)
# set up matplotlib style
# plt.style.use('seaborn-whitegrid')
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
    'worst_generalization_performance': '$\qquad\qquad\qquad\quad$  Generalization \n $\qquad\qquad\qquad\quad$ performance\n',
    'performance': '$\qquad\qquad\qquad\quad$ Nominal\n $\qquad\qquad\qquad\quad$ performance\n',
    'inference_time': 'Inference\ntime\n\n',
    'model_complexity': 'Model                \nknowledge                ',
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
}

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
    
    for key in data.keys():
        
        temp_value =  (np.array(data[key]) - min_values[key]) \
                / (max_values[key] - min_values[key])
        temp_value = (1 - temp_value) if key in inverted_axes_name else temp_value
        temp_value = np.clip(temp_value, 0, 1)  # clip to [0, 1]
        normalized_data[key] = temp_value + lower_padding

    # normalized ID numbers
    # ID_normalized = {key: np.array(value) / max_values[key] + lower_padding for key, value in ID_numbers.items()}
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
        # Invert the values to have the higher values in the center

        values += values[:1]  # Close the plot for a better look
        # values = 1 - np.array(values) 
        if model_name in ['MAX', 'MIN']:
            # ax.plot(angles, values, color=plot_colors[model_name], )
            # ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            # ax.fill(angles, values, alpha=1, color=plot_colors[model_name], )
            continue
        else:
            # print(f'{values=}')
            ax.plot(angles, values, label=model_name, color=plot_colors[model_name], )
            ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            ax.fill(angles, values, alpha=0.15, color=plot_colors[model_name], )

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
            
    # add additional text for robustness axes
    ax.text(angles[metric_index['robustness_obs']], 1.75, 'Robustness', size=small_text_size)
    # ax.fill(angles, 1.5*np.ones(num_axis + 1), alpha=0.7, color='lightgray')
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks, fontsize=axis_label_fontsize)
    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.2), fontsize=text_fontsize)
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
metrics_data['GP-MPC']['worst_generalization_performance'] = 0.025 # max(transfer_metric['GP-MPC']['rmse'][0], transfer_metric['GP-MPC']['rmse'][-1])
metrics_data['GP-MPC']['performance'] = 0.019  # Manually set
metrics_data['GP-MPC']['inference_time'] = transfer_metric['GP-MPC']['inference_time']
metrics_data['GP-MPC']['model_complexity'] = 1
metrics_data['GP-MPC']['sampling_complexity'] = 660
metrics_data['GP-MPC']['robustness_proc'] = 4
metrics_data['GP-MPC']['robustness_obs'] = 70
metrics_data['GP-MPC']['robustness_param'] = 4.8

# Linear MPC
metrics_data['Linear MPC']['worst_generalization_performance'] = max(transfer_metric['Linear MPC']['rmse'][0], transfer_metric['Linear MPC']['rmse'][-1])
metrics_data['Linear MPC']['performance'] = transfer_metric['Linear MPC']['rmse'][2]
metrics_data['Linear MPC']['inference_time'] = transfer_metric['Linear MPC']['inference_time']
metrics_data['Linear MPC']['model_complexity'] = 2
metrics_data['Linear MPC']['sampling_complexity'] = 1
metrics_data['Linear MPC']['robustness_proc'] = 10
metrics_data['Linear MPC']['robustness_obs'] = 120
metrics_data['Linear MPC']['robustness_param'] = 4.8

# Nonlinear MPC
metrics_data['Nonlinear MPC']['worst_generalization_performance'] = max(transfer_metric['Nonlinear MPC']['rmse'][0], transfer_metric['Nonlinear MPC']['rmse'][-1])
metrics_data['Nonlinear MPC']['performance'] = transfer_metric['Nonlinear MPC']['rmse'][2]
metrics_data['Nonlinear MPC']['inference_time'] = transfer_metric['Nonlinear MPC']['inference_time']
metrics_data['Nonlinear MPC']['model_complexity'] = 0
metrics_data['Nonlinear MPC']['sampling_complexity'] = 1
metrics_data['Nonlinear MPC']['robustness_proc'] = 2
metrics_data['Nonlinear MPC']['robustness_obs'] = 60
metrics_data['Nonlinear MPC']['robustness_param'] = 1.4

# F-MPC
metrics_data['F-MPC']['worst_generalization_performance'] = max(transfer_metric['F-MPC']['rmse'][0], transfer_metric['F-MPC']['rmse'][-1])
metrics_data['F-MPC']['performance'] = transfer_metric['F-MPC']['rmse'][2]
metrics_data['F-MPC']['inference_time'] = transfer_metric['F-MPC']['inference_time']
metrics_data['F-MPC']['model_complexity'] = 0
metrics_data['F-MPC']['sampling_complexity'] = 1
metrics_data['F-MPC']['robustness_proc'] = 2
metrics_data['F-MPC']['robustness_obs'] = 35
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
metrics_data['PPO-MPC']['inference_time'] = 2.25e-3
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
metrics_data['PID']['robustness_proc'] = 10
metrics_data['PID']['robustness_obs'] = 120
metrics_data['PID']['robustness_param'] = 4.8

# iLQR
metrics_data['iLQR']['worst_generalization_performance'] = max(transfer_metric['iLQR']['rmse'][0], transfer_metric['iLQR']['rmse'][-1])
metrics_data['iLQR']['performance'] = transfer_metric['iLQR']['rmse'][2]
metrics_data['iLQR']['inference_time'] = transfer_metric['iLQR']['inference_time']
metrics_data['iLQR']['model_complexity'] = 0
metrics_data['iLQR']['sampling_complexity'] = 1
metrics_data['iLQR']['robustness_proc'] = 2
metrics_data['iLQR']['robustness_obs'] = 40
metrics_data['iLQR']['robustness_param'] = 1.2

# LQR
metrics_data['LQR']['worst_generalization_performance'] = max(transfer_metric['LQR']['rmse'][0], transfer_metric['LQR']['rmse'][-1])
metrics_data['LQR']['performance'] = transfer_metric['LQR']['rmse'][2]
metrics_data['LQR']['inference_time'] = transfer_metric['LQR']['inference_time']
metrics_data['LQR']['model_complexity'] = 2
metrics_data['LQR']['sampling_complexity'] = 1
metrics_data['LQR']['robustness_proc'] = 20
metrics_data['LQR']['robustness_obs'] = 120
metrics_data['LQR']['robustness_param'] = 4.2

max_values = {key: max([metrics_data[method][key] for method in metrics_data.keys()]) for key in metrics_data['GP-MPC'].keys()}
min_values = {key: min([metrics_data[method][key] for method in metrics_data.keys()]) for key in metrics_data['GP-MPC'].keys()}

# handtune max and min to make the plot look better
max_values['performance'] = 0.05
max_values['worst_generalization_performance'] = 0.2
max_values['robustness_proc'] = 15
max_values['inference_time'] =1.7e-3

# append the max and min values to the data
# read the argv
if len(sys.argv) > 1:
    # masks_algo = [int(i) for i in sys.argv[1:]]
    algo = sys.argv[1]
else:
    algo = 'pid'
    
if algo in tag_ctrl_list.values():
    algo = list(tag_ctrl_list.keys())[list(tag_ctrl_list.values()).index(algo)]
    
assert algo in tag_ctrl_list.keys(), f'Algorithm {algo} not found in tag_ctrl_list. Available algorithms: {list(tag_ctrl_list.keys())}'

data = {key: 0 for key in metrics_data['GP-MPC'].keys()}
for key in metrics_data['GP-MPC'].keys():
    data[key] = [metrics_data[algo][key]]
    # data[key].append(max_values[key])
    # data[key].append(min_values[key])
# algos = [algo, 'MAX', 'MIN']
algos = [algo]
spider(
    pd.DataFrame({
        'x': algos,
        # '$\qquad\qquad\qquad\quad$  Generalization \n $\qquad\qquad\qquad\quad$ performance\n':
        # 'Generalization \n performance\n':
        'worst_generalization_performance':
            data['worst_generalization_performance'],
        # '$\qquad\qquad\qquad\quad$ Nominal\n $\qquad\qquad\qquad\quad$ performance\n':
        'performance':
            data['performance'],
        # 'Inference\ntime\n\n':
        'inference_time':
            data['inference_time'],
        # 'Model                \nknowledge                ':
        'model_complexity':
            data['model_complexity'],
        # '\n\n\nSampling\ncomplexity':
        'sampling_complexity':
            data['sampling_complexity'],
        # 'P':
        'robustness_proc':
            data['robustness_proc'],
        # 'O':
        'robustness_obs':
            data['robustness_obs'],
        # r'$\theta$':
        'robustness_param':
            data['robustness_param'],
    }),
    id_column='x',
    title=None,
    lower_padding=0.15,
    plt_name=algo,
    max_values=max_values,
    min_values=min_values,
)
