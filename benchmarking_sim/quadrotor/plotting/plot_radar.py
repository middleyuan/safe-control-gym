import os
import sys

# import seaborn
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

script_dir = os.path.dirname(__file__)
def load_metric(transfer_metric, method):
    ctrl = tag_ctrl_list[method]
    res = np.load(
        f'{script_dir}/../data/{ctrl}_gen_results.npy', allow_pickle=True).item()
    transfer_metric[method] = {'rmse': [], 'rmse_std': [], 'inference_time': []}
    for T in episode_len_list:
        T = '_'+str(T)
        transfer_metric[method]['rmse'].append(res[T]['mean_rmse'])
        transfer_metric[method]['rmse_std'].append(res[T]['std_rmse'])
    transfer_metric[method]['rmse'] = np.array(transfer_metric[method]['rmse'])
    transfer_metric[method]['rmse_std'] = np.array(transfer_metric[method]['rmse_std'])
    transfer_metric[method]['inference_time'] = np.mean(res['inference_time'])
    return transfer_metric

# get the pyplot default color wheel
prop_cycle = plt.rcParams['axes.prop_cycle']

# set up Nature sytle plotting
# set up seaborn style
# seaborn.set(style='whitegrid', palette='deep')
# set up matplotlib style
# plt.style.use('seaborn-whitegrid')
# plt.rcParams.update({
#     # 'font.family': 'arial',
#     'grid.alpha': 0.3,
#     'savefig.bbox': 'tight',
#     'savefig.transparent': True,
#     # 'font.size': 20,
# })

SYS = 'quadrotor_2D_attitude'
tag_ctrl_list = {
    'iLQR': 'ilqr',
    'LQR': 'lqr',
    'PID': 'pid',
    'Linear MPC': 'linear_mpc_acados',
    'Nonlinear MPC': 'mpc_acados',
    'F-MPC': 'fmpc',
    'GP-MPC': 'gpmpc_acados_TP' if SYS == 'quadrotor_2D_attitude' else 'gpmpc_acados_TRP',
    'PPO': 'ppo',
    'SAC': 'sac',
    'DPPO': 'dppo',
    'PPO-MPC': 'ppo_mpc',
    'PPO-ID': 'ppo_id',
    'SAC-ID': 'sac_id',
    'DPPO-ID': 'dppo_id',
}
transfer_metric = {}
episode_len_list = [9, 10, 11, 12, 13, 14, 15]
transfer_metric = load_metric(transfer_metric, 'iLQR')
transfer_metric = load_metric(transfer_metric, 'F-MPC')
transfer_metric = load_metric(transfer_metric, 'Nonlinear MPC')
transfer_metric = load_metric(transfer_metric, 'Linear MPC')
transfer_metric = load_metric(transfer_metric, 'PID')
transfer_metric = load_metric(transfer_metric, 'LQR')
transfer_metric = load_metric(transfer_metric, 'GP-MPC')

plot_colors = {
    'GP-MPC': 'royalblue',
    'PPO': 'darkorange',
    'SAC': 'red',
    'DPPO': 'tab:pink',
    'PID': 'darkgray',
    'Linear MPC': 'green',
    'Nonlinear MPC': 'cadetblue',
    'F-MPC': 'darkblue',
    "iLQR": "slateblue",
    'LQR': 'blueviolet',
    'PPO-MPC': 'tan',
    # 'PPO-ID': 'orange',
    # 'SAC-ID': 'red',
    # 'DPPO-ID': 'tab:pink',
    'MAX': 'none',
    'MIN': 'none',
}

axis_label_fontsize = 20
text_fontsize = 30
supertitle_fontsize = 30
subtitle_fontsize = 30
small_text_size = 20

metric_index = {
    # 'fast': 0,
    'worst_generalization_performance': 0,
    'performance': 1,
    'inference_time': 2,
    'model_complexity': 3,
    'sampling_complexity': 4,
    'robustness_proc': 5,
    'robustness_obs': 6,
    'robustness_param': 7,
    # 'slow': 8,
}

inverted_axes_name = [
    # 'fast',
    'worst_generalization_performance',
    'performance',
    'inference_time',
    'model_complexity',
    'sampling_complexity',
    # 'slow',    
]
inverted_axes_index = [metric_index[i] for i in inverted_axes_name]

ID_numbers = {
    'SAC':{
    'robustness_proc': 12,
    'robustness_obs': 100,
    'robustness_param': 5,
},
    'PPO':{
    'robustness_proc': 20,
    'robustness_obs': 80,
    'robustness_param': 4,
},
    'DPPO':{
    'robustness_proc':35,
    'robustness_obs': 90,
    'robustness_param': 5,
},
    # 'PPO-MPC':{
    # 'robustness_proc': 0,
    # 'robustness_obs': 1,
    # 'robustness_param': 2,
}

def spider(df, *, id_column, title=None, subtitle=None, max_values=None, padding=1.25, plt_name=''):
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient='list')
    ids = df[id_column].tolist()

    lower_padding = (padding - 1) / 2
    # upper_padding = 1 + lower_padding * 2
    upper_padding = 1 + 7 * lower_padding
    # upper_padding = 1.05

    if max_values is None:
        max_values = {key: upper_padding * max(value) for key, value in data.items()}
    # else:
    #     # make the max values into a dict
    #     max_values = {key: upper_padding * value for key, value in zip(categories, max_values)}
    # if max_values is None:
    #     max_values = {}
    #     for idx, key in enumerate(data.keys()):
    #         if idx in inverted_axes_index:
    #             print('invert', key)
    #             max_values[key] = upper_padding * min(data[key])
    #         else:
    #             max_values[key] = upper_padding * max(data[key])
    # else:
    #     max_values = {key: upper_padding * max(value) for key, value in data.items()}

    normalized_data = {key: np.array(value) / max_values[key] + lower_padding for key, value in data.items()}

    # normalized ID numbers
    # ID_normalized = {key: np.array(value) / max_values[key] + lower_padding for key, value in ID_numbers.items()}
    num_vars = len(data.keys()) # number of axes
    tiks = list(data.keys())
    tiks += tiks[:1]
    # print('tiks:', tiks)
    # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    # start_angle = np.deg2rad(7.5)
    start_angle = np.deg2rad(0)
    angles = [
        # start_angle, # faster 
        start_angle, # worst generalization
        np.deg2rad(60), # performance
        np.deg2rad(120), # inference time
        np.deg2rad(180), # model complexity
        np.deg2rad(240), # sampling complexity
        np.deg2rad(300 - 15), # robustness proc
        np.deg2rad(300), # observation noise
        np.deg2rad(300 + 15), # robustness param
        # np.deg2rad(360) - start_angle, # slower
        start_angle, # close the circle
    ]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True), )
    # ax.set_theta_offset(np.deg2rad(60))  # Rotate by 60 degrees
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]

        # Invert the values to have the higher values in the center
        for j in inverted_axes_index:
            values[j] = 1 - np.array(values[j])

        values += values[:1]  # Close the plot for a better look
        # values = 1 - np.array(values) 
        if model_name in ['MAX', 'MIN']:
            ax.plot(angles, values, color=plot_colors[model_name], )
            ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            ax.fill(angles, values, alpha=0.15, color=plot_colors[model_name], )
            continue
        else:
            ax.plot(angles, values, label=model_name, color=plot_colors[model_name], )
            ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            ax.fill(angles, values, alpha=0.15, color=plot_colors[model_name], )

        for _x, _y, t in zip(angles, values, actual_values):
            # customize the text
            if _x == angles[metric_index['inference_time']]:
                t = f'{t:.1E}' if isinstance(t, float) else str(t)
            elif _x == angles[metric_index['sampling_complexity']]:
                t = '0' if t == int(1) else f'{t:.1E}' # write number in scientific notation
            elif _x == angles[metric_index['model_complexity']]:
                if t == 1: t = 'Model-free'
                if t == 40: t = '   Linear\n   model'
                if t == 80: t = '   Partially uncertain \n nonlinear model'
                if model_name == 'PID': t = '   Kinematic \n   model'
                if t == 120: t = 'Perfect nonlinear\n   model'
            elif _x == angles[metric_index['robustness_param']]:
                t = f'{t:.1f}' if isinstance(t, float) else str(t)
            else:
                t = f'{t:.3f}' if isinstance(t, float) else str(t)

            # t = t.center(10, ' ')
            # cusotmize the text position for axes
            # if _x in [angles[metric_index['slow']]]:
            #     ax.text(_x - 0.15, _y - 0.1, t, size=small_text_size)
            # elif _x in [angles[metric_index['fast']]]:
            #     ax.text(_x + 0.1, _y - 0.1, t, size=small_text_size)
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
            
            # # plot extra dot for ID numbers
            # if model_name in ID_normalized.keys():
            #     if _x == angles[metric_index['robustness_proc']]:
            #         # plot a dot for the ID number
            #         ax.scatter(_x, ID_normalized[model_name]['robustness_proc'],
            #                 facecolor=plot_colors[model_name], s=100, edgecolor='black', linewidth=1)
            #     if _x == angles[metric_index['robustness_obs']]:
            #         ax.scatter(_x, ID_numbers[model_name]['robustness_obs'],
            #                 facecolor=plot_colors[model_name], s=100, edgecolor='black', linewidth=1)
            #     if _x == angles[metric_index['robustness_param']]:
            #         ax.scatter(_x, ID_numbers[model_name]['robustness_param'],
            #                 facecolor=plot_colors[model_name], s=100, edgecolor='black', linewidth=1)
    
    # add additional text for robustness axes
    ax.text(angles[metric_index['robustness_obs']], 1.25, 'Robustness', size=small_text_size)
    # ax.text(0, 1.35, 'Generalization', size=small_text_size)
    
    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05, color='lightgray')
    # ax.fill(angles[0:3], np.ones(3), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks, fontsize=axis_label_fontsize)
    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.2), fontsize=text_fontsize)
    if title is not None: plt.suptitle(title, fontsize=supertitle_fontsize)
    if subtitle is not None: plt.title(subtitle, fontsize=subtitle_fontsize)
    fig_save_path = os.path.join(script_dir, f'radar/radar_{plt_name}.pdf')
    fig.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    fig.savefig(fig_save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'figure saved as {fig_save_path}')

radar = spider
slow_performance = [transfer_metric['GP-MPC']['rmse'][-1],  # GP-MPC
                   transfer_metric['Linear MPC']['rmse'][-1], # Linear-MPC
                   transfer_metric['Nonlinear MPC']['rmse'][-1],  # MPC
                   transfer_metric['F-MPC']['rmse'][-1], # F-MPC
                   0.107995445,  # PPO
                   0.08379055,  # SAC
                   0.13267572,  # DPPO
                   0.01077949, # PPO-MPC
                   transfer_metric['PID']['rmse'][-1], # PID
                   transfer_metric['iLQR']['rmse'][-1], # iLQR
                   transfer_metric['LQR']['rmse'][-1], # LQR
                #    0.12597461,  # PPO-ID
                #    0.26743613,  # SAC-ID
                #    0.00123366,  # DPPO-ID
]

fast_performance = [
    transfer_metric['GP-MPC']['rmse'][0],
    transfer_metric['Linear MPC']['rmse'][0],
    transfer_metric['Nonlinear MPC']['rmse'][0],
    transfer_metric['F-MPC']['rmse'][0],
    0.11419083, # PPO
    0.20906559, # SAC
    0.14269699, # DPPO
    0.04358512, # PPO-MPC
    transfer_metric['PID']['rmse'][0], 
    transfer_metric['iLQR']['rmse'][0], 
    transfer_metric['LQR']['rmse'][0], 
    # 0.1367247, # PPO-ID
    # 0.26879758, # SAC-ID
    # 0.20101124, # DPPO-ID
]

worst_generalization_performance = [
    max(i, j) for i, j in zip(fast_performance, slow_performance)
]
print('worst_generalization_performance:', worst_generalization_performance)

performance = [transfer_metric['GP-MPC']['rmse'][2],   # GP-MPC
               transfer_metric['Linear MPC']['rmse'][2],   # Linear-MPC
               transfer_metric['Nonlinear MPC']['rmse'][2],   # MPC
               transfer_metric['F-MPC']['rmse'][2],   # F-MPC
               0.012809196573431799,  # PPO
               0.032350691213897304,  # SAC
               0.021298943332121172,  # DPPO
               0.01362814727788425, # PPO-MPC
               transfer_metric['PID']['rmse'][2], # PID
               transfer_metric['iLQR']['rmse'][2], # iLQR
               transfer_metric['LQR']['rmse'][2], # LQR
                # 0.012809196573431799,  # PPO-ID
                # 0.032350691213897304,  # SAC-ID
                # 0.021298943332121172,  # DPPO-ID
               ]
inference_time = [transfer_metric['GP-MPC']['inference_time'],  # GP-MPC
                  transfer_metric['Linear MPC']['inference_time'],  # Linear-MPC
                  transfer_metric['Nonlinear MPC']['inference_time'],  # MPC
                  transfer_metric['F-MPC']['inference_time'],  # F-MPC
                  7.31e-5, # PPO
                  8.72e-5, # SAC
                  7.28e-5, # DPPO
                  2.25e-3, # PPO-MPC
                  transfer_metric['PID']['inference_time'],  # PID
                  transfer_metric['iLQR']['inference_time'],  # iLQR
                  transfer_metric['LQR']['inference_time'],  # LQR
                    # 7.31e-5, # PPO-ID
                    # 8.72e-5, # SAC-ID
                    # 7.28e-5, # DPPO-ID

                  ]
model_complexity = [80, # GP-MPC
                    40, # Linear-MPC
                    120, # MPC
                    120, # F-MPC
                    1, # PPO
                    1, # SAC
                    1, # DPPO
                    80, # PPO-MPC
                    80, # PID
                    120, # iLQR
                    40, # LQR 
                    # 1, # PPO-ID
                    # 1, # SAC-ID
                    # 1, # DPPO-ID
                    ]
sampling_complexity = [ int(660),
                        int(1),
                        int(1),
                        int(1),
                        int(1e6), # PPO
                        int(0.8e6), # SAC
                        int(1e6), # DPPO
                        int(0.4e6), # PPO-MPC
                        int(1),
                        int(1),
                        int(1),
                        # int(1e6), # PPO-ID
                        # int(0.8e6), # SAC-ID
                        # int(1e6), # DPPO-ID
                       ]
robustness_proc = [ 4, # GP-MPC
                    10, # Linear-MPC
                    2, # MPC
                    3, # F-MPC
                    4, # PPO
                    6, # SAC
                    7, # DPPO
                    3, # PPO-MPC
                    10, # PID
                    2, # iLQR
                    20, # LQR
                    # ID_numbers['PPO']['robustness_proc'], # PPO-ID
                    # ID_numbers['SAC']['robustness_proc'], # SAC-ID
                    # ID_numbers['DPPO']['robustness_proc'], # DPPO-ID
              ]

robustness_obs = [
    70, # GP-MPC
    120, # Linear-MPC
    60, # MPC
    100, # F-MPC
    18, # PPO
    100, # SAC
    25, # DPPO
    45, # PPO-MPC
    120, # PID
    40, # iLQR
    120, # LQR   
    # ID_numbers['PPO']['robustness_obs'], # PPO-ID
    # ID_numbers['SAC']['robustness_obs'], # SAC-ID
    # ID_numbers['DPPO']['robustness_obs'], # DPPO-ID
]

robustness_param = [
    4.8, # GP-MPC
    4.8, # Linear-MPC
    1.4, # MPC
    2.8, # F-MPC
    1.2, # PPO
    2.0, # SAC
    1.8, # DPPO
    2.8, # PPO-MPC
    4.8, # PID
    1.2, # iLQR
    4.8, # LQR
    # ID_numbers['PPO']['robustness_param'], # PPO-ID
    # ID_numbers['SAC']['robustness_param'], # SAC-ID
    # ID_numbers['DPPO']['robustness_param'], # DPPO-ID
]

data = [
    # fast_performance,
    worst_generalization_performance,
    performance, 
    inference_time, 
    model_complexity, 
    sampling_complexity, 
    robustness_proc,
    robustness_obs,
    robustness_param,
    # slow_performance, 
]

# get the max and min values (NOTE: the axis is inverted)
# max_values = [min(i) for i in data]
min_values = []
max_values = []
for i in metric_index.values():
    if i in inverted_axes_index:
        min_values.append(max(data[i]))
        max_values.append(min(data[i]))
    else:
        min_values.append(min(data[i]))
        max_values.append(max(data[i]))
# min_values = [max(i) for i in data]
# manually tune some axes
# min_values[metric_index['fast']] = 0.25 # gen performance fast
# min_values[metric_index['performance']] = 0.0
max_values[metric_index['performance']] = 0.05
min_values[metric_index['performance']] = 0.005
max_values[metric_index['worst_generalization_performance']] = 0.2
min_values[metric_index['worst_generalization_performance']] = 0.033
min_values[metric_index['inference_time']] = 1.0e-3
# max_values[metric_index['robustness_proc']] = 50
max_values[metric_index['robustness_obs']] = 80
max_values[metric_index['robustness_proc']] = 20
min_values[metric_index['robustness_proc']] = 2

max_values[metric_index['robustness_param']] = 5.0

# append the max and min values to the data (but only plot empty)
for i, d in enumerate(data): 
    data[i].append(max_values[i])
    data[i].append(min_values[i])

# append the max and min values to the data
algos = ['GP-MPC',
         'Linear MPC',
         'Nonlinear MPC',
         'F-MPC',
         'PPO',
         'SAC',
         'DPPO',
         'PPO-MPC',
         'PID',
         'iLQR',
         'LQR',
         'PPO-ID',
         'SAC-ID',
         'DPPO-ID',
         'MAX', 'MIN']

# read the argv
if len(sys.argv) > 1:
    # masks_algo = [int(i) for i in sys.argv[1:]]
    algo = sys.argv[1]
    if algo == 'GP-MPC':
        masks_algo = [0]
    elif algo == 'PPO':
        masks_algo = [4]
    elif algo == 'SAC':
        masks_algo = [5]
    elif algo == 'DPPO':
        masks_algo = [6]
    elif algo == 'PPO-MPC':
        masks_algo = [7]
    elif algo == 'PID':
        masks_algo = [8]
    elif algo == 'iLQR':
        masks_algo = [9]
    elif algo == 'LQR':
        masks_algo = [10]
    elif algo == 'F-MPC':
        masks_algo = [3]
    elif algo == 'Nonlinear-MPC':
        masks_algo = [2]
    elif algo == 'Linear-MPC':
        masks_algo = [1 ]
    elif algo == 'PPO-ID':
        masks_algo = [11]
    elif algo == 'SAC-ID':
        masks_algo = [12]
    elif algo == 'DPPO-ID':
        masks_algo = [13]
else:
    masks_algo = [8] # PID
masks_algo += [-2, -1]  # add MAX and MIN

data = np.array(data)[:, masks_algo]
data = data.tolist()
algos = [algos[i] for i in masks_algo]
print(algos)

spider(
    pd.DataFrame({
        # 'x': [*'ab'],
        'x': algos,
        # '$\qquad\qquad\qquad\quad$  Fast\n $\qquad\qquad\qquad\quad$ performance\n':
        # '$\quad$  Fast':
        #     data[metric_index['fast']],
        '$\qquad\qquad\qquad\quad$  Generalization \n $\qquad\qquad\qquad\quad$ performance\n':
            data[metric_index['worst_generalization_performance']],
        '$\qquad\qquad\qquad\quad$ Nominal\n $\qquad\qquad\qquad\quad$ performance\n':
            data[metric_index['performance']],
        'Inference\ntime\n\n':
            data[metric_index['inference_time']],
        'Model                \nknowledge                ':
            [int(data[metric_index['model_complexity']][i]) for i in range(len(data[3]))],
        '\n\n\nSampling\ncomplexity':
            data[metric_index['sampling_complexity']],
        # '\n\nProcess noise':
        # '$\qquad$ Proc':
        'P':
            [int(data[metric_index['robustness_proc']][i]) for i in range(len(data[5]))],
        # '\n\nObservation noise':
        # '$\qquad$ Obs':
        'O':
            [int(data[metric_index['robustness_obs']][i]) for i in range(len(data[6]))],
        r'$\theta$':
        # '$\qquad$ Param':
            data[metric_index['robustness_param']],
        # '$\qquad\qquad\qquad\quad$  Slow\n $\qquad\qquad\qquad\quad$ performance\n\n':
        # '$\quad$ Slow':
        #     data[metric_index['slow']],
        # '\n\nParameter noise':
            # [data[7][i] for i in range(len(data[7]))],
        # '\n\nRobustness\n(process)':
        #     [int(data[5][i]) for i in range(len(data[5]))],
        # '\n\nRobustness\n(observation)':
        #     [int(data[6][i]) for i in range(len(data[6]))],
        # '\n\nRobustness\n(parameter)':
        #     [int(data[7][i]) for i in range(len(data[7]))],
    }),

    id_column='x',
    # title='   Overall Comparison',
    # title = algos[0],
    title=None,
    # subtitle='(Normalized linear scale)',
    padding=1.1,
    # padding=1,
    plt_name=algos[0],
    # max_values=max_values,
)
