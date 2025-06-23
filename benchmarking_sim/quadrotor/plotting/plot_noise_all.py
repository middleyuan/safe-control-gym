import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# script dir
script_dir = os.path.dirname(os.path.abspath(__file__))

max_seed = 10
metric_name = 'metrics.txt'
s = 2 # times std

plot_colors = {
    'GP-MPC': 'royalblue',
    'PPO': 'darkorange',
    'SAC': 'red',
    'DPPO': 'pink',
    'iLQR': 'darkgray',
    'Linear MPC': 'green',
    'Nonlinear MPC': 'cadetblue',
    "PID": "tab:gray",
    "iLQR": "slateblue",
    'LQR': 'blueviolet',
    "F-MPC": "darkblue",
    'MAX': 'none',
    'MIN': 'none',
}


if len(sys.argv) > 1:
    noise_option = sys.argv[1]
else:
    noise_option = 'obs_noise'
    # noise_option = 'proc_noise'
    # noise_option = 'param'
print('noise_option', noise_option)

linear_mpc_data = np.load(f'{script_dir}/../data/linear_mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
mpc_data = np.load(f'{script_dir}/../data/mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
gpmpc_data = np.load(f'{script_dir}/../data/gpmpc_acados_TP_{noise_option}_results.npy', allow_pickle=True).item()
ilqr_data = np.load(f'{script_dir}/../data/ilqr_{noise_option}_results.npy', allow_pickle=True).item()
lqr_data = np.load(f'{script_dir}/../data/lqr_{noise_option}_results.npy', allow_pickle=True).item()
pid_data = np.load(f'{script_dir}/../data/pid_{noise_option}_results.npy', allow_pickle=True).item()
fmpc_data = np.load(f'{script_dir}/../data/fmpc_{noise_option}_results.npy', allow_pickle=True).item()



noise_data = {
              'Linear MPC': linear_mpc_data, 
              'Nonlinear MPC': mpc_data, 
              'GP-MPC': gpmpc_data,
              'iLQR': ilqr_data,
              'LQR': lqr_data,
              'PID': pid_data,
              'F-MPC': fmpc_data,
              } 
if noise_option in ['obs_noise']:
    noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,\
                         12, 14, 16, 18, 20, 25, 30, 35, 40, \
                         45, 50, 60, 70, 80, 90, 100]
elif noise_option == 'proc_noise':
    noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,\
                    12, 14, 16, 18, 20, 25, 30, 35, 40, \
                    45, 50, 60, 70, 80, 90, 100]
elif noise_option == 'param':
    noise_scale = [0, 0.01, 0.02, 0.05, 0.1, \
                     0.2, 0.4, 0.6, 0.8, 1.0, 1.2, \
                     1.4, 1.6, 1.8, 2.0, 2.2, 2.4, \
                     2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0]
    noise_scale.sort()

fig = plt.figure(figsize=(8, 3))

# plot rmse degradation for all methods
double_results = {}

for method in noise_data.keys():
    print(method)
    plt.plot(noise_scale, noise_data[method]['rmse_degradation_mean'], label=method, color=plot_colors[method])
    plt.fill_between(noise_scale, 
                     noise_data[method]['rmse_degradation_mean']-s*noise_data[method]['rmse_degradation_std'],  
                     noise_data[method]['rmse_degradation_mean']+s*noise_data[method]['rmse_degradation_std'], color=plot_colors[method], alpha=0.1)
    # print the noise scale that the rmse degradation is larger than 200
    for i, scale in enumerate(noise_scale):
        if noise_data[method]['rmse_degradation_mean'][i] > 200:
            print(f"Method {method} reaches 200% performance at noise scale {scale}")
            double_results[method] =  scale
            break
    
plt.xlabel("Noise Scale")
plt.ylabel("Relative Performance %")
if noise_option == 'obs_noise':
    plt.legend(ncol=2)
    plt.ylim(0, 700)
    plt.xlim(0, 100)
    plt.title("Performance Degradation with Observation Noise")
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
    # plt.text(100, 310, 'Nonlinear MPC 200%')
    # plt.text(70, 240, 'GP-MPC 200%')
    # plt.text(50, 300, 'iLQR 200%')
    # plt.text(100, 370, 'F-MPC 200%')
elif noise_option == 'proc_noise':
    plt.legend(ncol=2, loc='upper right')
    plt.title("Performance Degradation with Process Noise")
    plt.plot(noise_scale, [200]*len(noise_scale), \
             color='grey', linestyle='-.', label='Relative Perf=200%')
    # plot vertical line at 200% performance
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
    # plt.text(11, 3700, 'Linear MPC 200%')
    # plt.text(6, 6100, 'Nonlinear MPC 200%')
    # plt.text(6, 5400, 'GP-MPC 200%')
    # plt.text(6, 300, 'iLQR 200%')
    # plt.text(16, 1400, 'LQR 200%')
    # plt.text(11, 3200, 'PID 200%')
    # plt.text(6, 6700, 'F-MPC 200%')
elif noise_option == 'param':
    plt.title("Performance Degradation with Parametric Uncertainty")
    plt.xlabel("Randomization scale")
    plt.ylim(0, 1000)
    plt.xlim(0, 5)
    plt.legend(ncol=2, loc='upper right')
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
    # plt.text(4.61, 100, 'Linear MPC 200%')
    # plt.text(3.21, 250, 'Nonlinear MPC 200%')
    # # plt.text(1.21, 250, 'GP-MPC 200%')
    # plt.text(3.01, 200, 'iLQR 200%')
    # # plt.text(16, 1400, 'LQR 200%')
    # plt.text(4.41, 200, 'PID 200%')
    # plt.text(2.81, 100, 'F-MPC 200%')
# plot an empty line for the legend
plt.plot(0, np.nan, linestyle='--', color='grey', label='Relative Perf=200%')

plt.plot(noise_scale, [200]*len(noise_scale), \
            color='grey', linestyle='-.', label='Relative Perf=200%')
plot_save_name = f"robustness_model-based_{noise_option}"
plot_save_name = os.path.join(script_dir, 'noise', f'{plot_save_name}.png')
plt.savefig(plot_save_name,bbox_inches="tight", pad_inches=0.1)
print(f"plots saved as {plot_save_name}")

# Absolute performance degradation
fig_abs = plt.figure(figsize=(8, 3))

abs_rmse_threshold = 0.1
abs_rmse_threshold_results = {}

for method in noise_data.keys():
    if 'rmse_mean' in noise_data[method] and 'rmse_std' in noise_data[method]:
        plt.plot(noise_scale, noise_data[method]['rmse_mean'], label=method, color=plot_colors[method])
        plt.fill_between(noise_scale,
                         noise_data[method]['rmse_mean'] - s * noise_data[method]['rmse_std'],
                         noise_data[method]['rmse_mean'] + s * noise_data[method]['rmse_std'],
                         color=plot_colors[method], alpha=0.1)
        # Check when absolute rmse exceeds threshold
        for i, scale in enumerate(noise_scale):
            if noise_data[method]['rmse_mean'][i] > abs_rmse_threshold:
                print(f"Method {method} exceeds absolute RMSE of {abs_rmse_threshold} at noise scale {scale}")
                abs_rmse_threshold_results[method] = scale
                break
    else:
        print(f"Warning: 'rmse_mean' or 'rmse_std' not found for method {method}. Skipping absolute plot.")

plt.xlabel("Noise Scale")
plt.ylabel("RMSE")
if noise_option == 'obs_noise':
    plt.legend(ncol=2)
    plt.xlim(0, 100)
    plt.ylim(0, 0.2)
    plt.title("Absolute Performance with Observation Noise")
    for method in abs_rmse_threshold_results.keys():
        plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'proc_noise':
    plt.legend(ncol=2, loc='upper left')
    plt.ylim(0, 1.5)
    plt.title("Absolute Performance with Process Noise")
    for method in abs_rmse_threshold_results.keys():
        plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'param':
    plt.title("Absolute Performance with Parametric Uncertainty")
    plt.xlabel("Randomization scale")
    plt.xlim(0, 5)
    plt.ylim(0, 0.4)
    plt.legend(ncol=2, loc='upper left')
    for method in abs_rmse_threshold_results.keys():
        plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])

plt.plot(noise_scale, [abs_rmse_threshold]*len(noise_scale), \
            color='grey', linestyle='-.', label=f'RMSE={abs_rmse_threshold}')

plot_save_name_abs = f"robustness_model-based_{noise_option}_absolute"
plot_save_name_abs = os.path.join(script_dir, 'noise', f'{plot_save_name_abs}.png')
plt.savefig(plot_save_name_abs, bbox_inches="tight", pad_inches=0.1)
print(f"plots saved as {plot_save_name_abs}")