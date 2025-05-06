import os
import sys

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

SYS = 'quadrotor_2D_attitude'
tag_ctrl_list = {
    'iLQR': 'ilqr',
    'LQR': 'lqr',
    'PID': 'pid',
    'Linear MPC': 'linear_mpc_acados',
    'Nonlinear MPC': 'mpc_acados',
    'F-MPC': 'fmpc',
    'GP-MPC': 'gpmpc_acados_TP' if SYS == 'quadrotor_2D_attitude' else 'gpmpc_acados_TRP',
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
                if t == 80: t = 'Nonlinear\n   model'
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
    fig_save_path = os.path.join(script_dir, f'/radar/radar_{plt_name}.pdf')
    fig.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    fig.savefig(fig_save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'figure saved as {fig_save_path}')

radar = spider
slow_performance = [transfer_metric['GP-MPC']['rmse'][-1],  # GP-MPC
                   transfer_metric['Linear MPC']['rmse'][-1], # Linear-MPC
                   transfer_metric['Nonlinear MPC']['rmse'][-1],  # MPC
                   transfer_metric['F-MPC']['rmse'][-1], # F-MPC
                   0.15347746,  # PPO
                   0.11081217,  # SAC
                   0.16864583,  # DPPO
                   transfer_metric['PID']['rmse'][-1], # PID
                   transfer_metric['iLQR']['rmse'][-1], # iLQR
                   transfer_metric['LQR']['rmse'][-1], # LQR
]

fast_performance = [
    transfer_metric['GP-MPC']['rmse'][0],
    transfer_metric['Linear MPC']['rmse'][0],
    transfer_metric['Nonlinear MPC']['rmse'][0],
    transfer_metric['F-MPC']['rmse'][0],
    0.1233724,
    0.12137596,
    0.13649782,
    transfer_metric['PID']['rmse'][0], 
    transfer_metric['iLQR']['rmse'][0], 
    transfer_metric['LQR']['rmse'][0], 
]

# worst_generalization_performance = [max(transfer_metric[method]['rmse']) \
#                                     for method in tag_ctrl_list.keys()] 
worst_generalization_performance = [
    max(i, j) for i, j in zip(fast_performance, slow_performance)
]
print('worst_generalization_performance:', worst_generalization_performance)

performance = [transfer_metric['GP-MPC']['rmse'][2],   # GP-MPC
               transfer_metric['Linear MPC']['rmse'][2],   # Linear-MPC
               transfer_metric['Nonlinear MPC']['rmse'][2],   # MPC
               transfer_metric['F-MPC']['rmse'][2],   # F-MPC
               0.021604097983791027,  # PPO
               0.04410137020535634,  # SAC
               0.03087288620530012,  # DPPO
               transfer_metric['PID']['rmse'][2], # PID
               transfer_metric['iLQR']['rmse'][2], # iLQR
               transfer_metric['LQR']['rmse'][2], # LQR
               ]
inference_time = [transfer_metric['GP-MPC']['inference_time'],  # GP-MPC
                  transfer_metric['Linear MPC']['inference_time'],  # Linear-MPC
                  transfer_metric['Nonlinear MPC']['inference_time'],  # MPC
                  transfer_metric['F-MPC']['inference_time'],  # F-MPC
                  7.31e-5, # PPO
                  8.72e-5, # SAC
                  7.28e-5, # DPPO
                  transfer_metric['PID']['inference_time'],  # PID
                  transfer_metric['iLQR']['inference_time'],  # iLQR
                  transfer_metric['LQR']['inference_time'],  # LQR
                  ]
model_complexity = [80, # GP-MPC
                    40, # Linear-MPC
                    80, # MPC
                    80, # F-MPC
                    1, # PPO
                    1, # SAC
                    1, # DPPO
                    1, # PID
                    80, # iLQR
                    40, # LQR 
                    ]
sampling_complexity = [ int(660),
                        int(1),
                        int(1),
                        int(1),
                        int(2.5 * 1e5), # PPO
                        int(1.32 * 1e5), # SAC
                        int(4.09 * 1e5), # DPPO
                        int(1),
                        int(1),
                        int(1),
                       ]
robustness_proc = [ 5, # GP-MPC
                    15, # Linear-MPC
                    4, # MPC
                    3, # F-MPC
                    4, # PPO
                    10, # SAC
                    3, # DPPO
                    15, # PID
                    4, # iLQR
                    15, # LQR
              ]

robustness_obs = [
    60, # GP-MPC
    120, # Linear-MPC
    100, # MPC
    100, # F-MPC
    15, # PPO
    100, # SAC
    15, # DPPO
    120, # PID
    50, # iLQR
    120, # LQR   
]

robustness_param = [
    5.1, # GP-MPC
    4.6, # Linear-MPC
    3.2, # MPC
    2.8, # F-MPC
    1.0, # PPO
    2.0, # SAC
    1.2, # DPPO
    4.4, # PID
    3.0, # iLQR
    5.1, # LQR
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
min_values[metric_index['performance']] = 0.06
# max_values[metric_index['performance']] = 0.02
# max_values[metric_index['robustness_proc']] = 50
max_values[metric_index['robustness_obs']] = 80
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
         'PID',
         'iLQR',
         'LQR',
         'MAX', 'MIN']

# read the argv
if len(sys.argv) > 1:
    # masks_algo = [int(i) for i in sys.argv[1:]]
    algo = sys.argv[1]
    # masks_algo.append(-2, -1)
    # masks_algo.append()
    if algo == 'GP-MPC':
        masks_algo = [0, -2, -1]
    elif algo == 'PPO':
        masks_algo = [4, -2, -1]
    elif algo == 'SAC':
        masks_algo = [5, -2, -1]
    elif algo == 'DPPO':
        masks_algo = [6, -2, -1]
    elif algo == 'PID':
        masks_algo = [7, -2, -1]
    elif algo == 'iLQR':
        masks_algo = [8, -2, -1]
    elif algo == 'LQR':
        masks_algo = [9, -2, -1]
    elif algo == 'F-MPC':
        masks_algo = [3, -2, -1]
    elif algo == 'Nonlinear-MPC':
        masks_algo = [2, -2, -1]
    elif algo == 'Linear-MPC':
        masks_algo = [1 , -2, -1]
else:
    masks_algo = [7,  -2, -1] # PID
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
