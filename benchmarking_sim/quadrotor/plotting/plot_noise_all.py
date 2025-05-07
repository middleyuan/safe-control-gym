import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# script dir
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)



max_seed = 3
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

# linear_mpc_data = np.load(f'{script_dir}/../data/linear_mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
mpc_data = np.load(f'{script_dir}/../data/mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
gpmpc_data = np.load(f'{script_dir}/../data/gpmpc_acados_TP_{noise_option}_results.npy', allow_pickle=True).item()
ilqr_data = np.load(f'{script_dir}/../data/ilqr_{noise_option}_results.npy', allow_pickle=True).item()
# lqr_data = np.load(f'{script_dir}/../data/lqr_{noise_option}_results.npy', allow_pickle=True).item()
# pid_data = np.load(f'{script_dir}/../data/pid_{noise_option}_results.npy', allow_pickle=True).item()
# fmpc_data = np.load(f'{script_dir}/../data/fmpc_{noise_option}_results.npy', allow_pickle=True).item()



noise_data = {
            #   'Linear MPC': linear_mpc_data, 
              'Nonlinear MPC': mpc_data, 
              'GP-MPC': gpmpc_data,
              'iLQR': ilqr_data,
            #   'LQR': lqr_data,
            #   'PID': pid_data,
            #   'F-MPC': fmpc_data,
              } 
if noise_option in ['obs_noise']:
    # noise_scale = [0,1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    noise_scale = [0,1,2,3,4,5,10,15,20,25,\
                     30,40,50,60,70,80,90,100]
elif noise_option == 'proc_noise':
    noise_scale = [0,1,2,3,4,5,10,15,20,25,\
                     30,40,50]
elif noise_option == 'param':
    noise_scale = np.arange(0, 5.0, 0.2)
    # noise_scale = mpc_data['0']['noise_factor']
    noise_scale.sort()
# print(len(noise_scale))

fig = plt.figure(figsize=(8, 3))

# plot rmse degradation for all methods
double_results = {}

for method in noise_data.keys():
    print(method)
    # print(len(noise_data[method]['rmse_degradation_mean']))

    plt.plot(noise_scale, noise_data[method]['rmse_degradation_mean'], label=method, color=plot_colors[method])
    plt.fill_between(noise_scale, 
                     noise_data[method]['rmse_degradation_mean']-noise_data[method]['rmse_degradation_std'],  
                     noise_data[method]['rmse_degradation_mean']+noise_data[method]['rmse_degradation_std'], color=plot_colors[method], alpha=0.1)
    # print the noise scale that the rmse degradation is larger than 200
    for i, scale in enumerate(noise_scale):
        if noise_data[method]['rmse_degradation_mean'][i] > 200:
            print(f"Method {method} 200 at noise scale {scale}")
            double_results[method] =  scale
            break
    
# plt.xscale("log")
# plt.gca().invert_xaxis()
# plt.yscale("log")
plt.xlabel("Noise Scale")
plt.ylabel("Relative Performance %")
if noise_option == 'obs_noise':
    plt.legend(ncol=2)
    plt.ylim(0, 500)
    plt.xlim(0, 100)
    plt.title("Robustness to observation noise")
    plt.plot(noise_scale, [200]*len(noise_scale), \
             color='grey', linestyle='-.', label='Relative Perf=200%')
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
    # plt.text(100, 310, 'Nonlinear MPC 200%')
    # plt.text(70, 240, 'GP-MPC 200%')
    # plt.text(50, 300, 'iLQR 200%')
    # plt.text(100, 370, 'F-MPC 200%')
elif noise_option == 'proc_noise':
    plt.legend(ncol=2, loc='upper right')
    plt.title("Robustness to process noise")
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
    plt.title("Robustness to parametric uncertainty")
    plt.xlabel("Randomization scale")
    plt.ylim(0, 500)
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

plt.savefig(f"robustness_model-based_{noise_option}.pdf",bbox_inches="tight", pad_inches=0.1)
plt.savefig(f"robustness_model-based_{noise_option}.png",bbox_inches="tight", pad_inches=0.1)