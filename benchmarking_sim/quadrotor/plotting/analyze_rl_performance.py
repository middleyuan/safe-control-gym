"""
RL Performance Analysis Script
Converted from data_analysis.ipynb

This script analyzes the performance of RL methods including:
- Task performance metrics
- Convergence analysis
- Generalization metrics
- Robustness analysis (observation noise, process noise, parametric noise, downwash)

Results are saved to:
- Plots: /home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/plotting/
- Data: /home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/
"""

import numpy as np 
import matplotlib.pyplot as plt
from collections import defaultdict
import glob 
import os
from pathlib import Path

# Import utilities
from benchmarking_sim.quadrotor.benchmark_util.utils import plot_colors, tag_ctrl_list, load_metric, STEPS_PER_SECOND

def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')
    
def load_from_log_file(path):
    '''Return x, y sequence data from the stat csv.'''
    with open(path, 'r') as f:
        lines = f.readlines()
    # Labels.
    xk, yk = [k.strip() for k in lines[0].strip().split(',')]
    # Values.
    x, y = [], []
    for line in lines[1:]:
        data = line.strip().split(',')
        x.append(float(data[0].strip()))
        y.append(float(data[1].strip()))
    x = np.array(x)
    y = np.array(y)
    return xk, x, yk, y

#### For reporting
def log(msg):
    log_messages.append(msg)
    print(msg)

def main():
    # Setup directories
    script_dir = Path(__file__).parent
    plot_dir = script_dir
    data_dir = script_dir.parent / 'data'
    convergence_dir = plot_dir / 'convergence'
    robustness_dir = plot_dir / 'robustness'
    
    # Create directories if they don't exist
    convergence_dir.mkdir(exist_ok=True)
    robustness_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Plot directory: {plot_dir}")

    # Plot colors and legends
    legends = {
        "ref": "black",
        "PPO": "PPO",
        "PPO3": "PPO with ilqr ref",
        "PPO4": "PPO with ilqr state ref",
        "SAC": "SAC",
        "TD3": "TD3",
        "DPPO": "DPPO",
        "GP-MPC": "GP-MPC",
        "PPO-MPC": "PPO-MPC"
    }

    #### Data info
    data_dir_path = str(data_dir)
    exp_name = "nominal"
    data_paths = {
        "PPO": data_dir_path + "/" + exp_name + "/quadrotor_2D_attitude_ppo_data",
        "SAC": data_dir_path + "/" + exp_name + "/quadrotor_2D_attitude_sac_data2",
        "DPPO": data_dir_path + "/" + exp_name + "/quadrotor_2D_attitude_dppo_data",
        "PPO-MPC": data_dir_path + "/" + exp_name + "/quadrotor_2D_attitude_ppo_mpc_data",
    }

    #### Seeds
    n_seeds = 10
    seeds = [i for i in range(0, n_seeds)] 
    global log_messages
    log_messages = []

    # =============================================================================
    # Performance Metrics
    # =============================================================================
    log("=== TASK PERFORMANCE ===")
    perf_metric = defaultdict(lambda: {'rmse': [], 'rmse_std': [], 'success': []})
    
    for method in data_paths.keys():
        for seed in seeds:
            path = data_paths[method] + "/seed"+ str(seed) +"_*/perf_metric.npy"
            file_path = glob.glob(path)
            if file_path:
                temp = np.load(file_path[0], allow_pickle=True).item()
                perf_metric[method]['rmse'].append(temp['rmse'])
                perf_metric[method]['rmse_std'].append(temp['rmse_std'])

    # Extract average inference time for each method
    inference_times = {}
    log("Inference times")
    for method in data_paths.keys():
        inference_time_list = []
        for seed in seeds:
            path = data_paths[method] + "/seed"+ str(seed) +"_*/perf_metric.npy"
            file_path = glob.glob(path)
            if file_path:
                temp = np.load(file_path[0], allow_pickle=True).item()
                if 'avarage_inference_time' in temp:
                    inference_time_list.append(temp['avarage_inference_time'][0])
                elif 'average_inference_time' in temp:
                    inference_time_list.append(temp['average_inference_time'][0])
        
        if inference_time_list:
            inference_times[method] = {
                'mean': np.array(inference_time_list).mean(),
                'std': np.array(inference_time_list).std()
            }
            log(f"Average inference time for {method}: {inference_times[method]['mean']:.6f} +/- {inference_times[method]['std']:.6f} seconds")
        else:
            inference_times[method] = {'mean': 0.0, 'std': 0.0}
            log(f"No inference time data found for {method}")

    log("\n")

    # =============================================================================
    # Convergence Analysis
    # =============================================================================
    log("=== CONVERGENCE ANALYSIS ===")
    
    perf_data = defaultdict(lambda: defaultdict())
    for method in data_paths.keys():
        print(method)
        for sd in os.listdir(data_paths[method]):
            if sd.startswith('seed'):
                seed = int(sd[4])  # Extract seed number
                base_path = os.path.join(data_paths[method], sd)
                
                xk, x, lk, l = load_from_log_file(base_path + "/logs/stat_eval/ep_length.log")
                xk, x, yk, y = load_from_log_file(base_path + "/logs/stat_eval/ep_return.log")
                xk, x, zk, z = load_from_log_file(base_path + "/logs/stat_eval/ep_return_std.log")
                xk, x, yk, m = load_from_log_file(base_path + "/logs/stat_eval/rmse.log")
                xk, x, yk, n = load_from_log_file(base_path + "/logs/stat_eval/rmse_std.log")
                perf_data[method].update({seed: {"data": x, "ep_return": y, "ep_return_std": z, "rmse": m, "rmse_std": n, "ep_length": l}})

    # Load GP-MPC data
    gp_name = '/gpmpc_acados_TP_hpo_convergence_results.npy'
    gp_mpc_data = np.load(str(data_dir) + gp_name, allow_pickle=True).item()
    log(f"GP-MPC final RMSE: {gp_mpc_data['rmse'][:, -1].mean()}")

    # Create convergence plot with time conversion
    eval_data = {}
    log("Sample efficiency")
    mean_fn = np.percentile
    perc = 80
    threshold, samples_till = 1.2, 2.0e6

    # Convert GP-MPC training steps to time
    gp_mpc_data['train_steps'] = gp_mpc_data['train_steps'] / STEPS_PER_SECOND

    fig = plt.figure(figsize=(8, 2)) 
    plt.axhline(xmin=0.0, xmax=1.95, y=0.1, linestyle='-.', color='k', label='rmse=0.1')

    plt.plot(gp_mpc_data['train_steps'], mean_fn(gp_mpc_data['rmse'], perc, axis=0), color=plot_colors["GP-MPC"], label='GP-MPC')
    plt.fill_between(gp_mpc_data['train_steps'], 
                    np.clip(mean_fn(gp_mpc_data['rmse'], perc, axis=0)-gp_mpc_data['rmse'].std(axis=0), 0, 10), 
                    np.clip(mean_fn(gp_mpc_data['rmse'], perc, axis=0)+gp_mpc_data['rmse'].std(axis=0), 0, 10), 
                    color=plot_colors["GP-MPC"], alpha=0.25)

    for t, method in enumerate(data_paths.keys()):
        print(method)
        temp = np.zeros((n_seeds, 6, perf_data[method][seeds[0]]["data"].shape[0]))
        for seed in seeds:
            if seed in perf_data[method]:
                temp[seed, 0, :] = perf_data[method][seed]["data"]
                temp[seed, 1, :] = perf_data[method][seed]["ep_return"]
                temp[seed, 2, :] = perf_data[method][seed]["ep_return_std"]
                temp[seed, 3, :] = perf_data[method][seed]["rmse"]
                temp[seed, 4, :] = perf_data[method][seed]["rmse_std"]
                temp[seed, 5, :] = perf_data[method][seed]["ep_length"]
        eval_data.update({method: temp})
        rmse_mean = mean_fn(temp[:,3,:], perc, axis=0)
        rmse_ref = rmse_mean[-1]
        mask = rmse_mean < threshold * rmse_ref
        if np.any(mask):
            log(f"For {method}, Threshold: {threshold * rmse_ref:.6f} with data axis: {temp[0,0, mask][0]}")

        # Convert horizontal axis to time
        h_axis = temp[0,0,:] + 1
        h_axis = h_axis / STEPS_PER_SECOND

        plt.plot(h_axis, rmse_mean, color=plot_colors[method], label=legends[method])
        plt.fill_between(h_axis, 
                         np.clip(mean_fn(temp[:,3,:]-temp[:,4,:], perc, axis=0), 0, 10),  
                         np.clip(mean_fn(temp[:,3,:]+temp[:,4,:], perc, axis=0), 0, 10), color=plot_colors[method], alpha=0.25)

    plt.legend(ncol=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training steps [s]")
    plt.ylabel("RMSE [m]")
    
    # Save convergence plot
    plt.savefig(convergence_dir / "convergence.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(convergence_dir / "convergence.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    log(f"Convergence plot saved to {convergence_dir}")

    log("\n")

    # =============================================================================
    # Generalization Analysis
    # =============================================================================
    log("=== GENERALIZATION ANALYSIS ===")
    
    episode_len_list = [9, 10, 11, 12, 13, 14, 15]
    episode_len_list2 = [4.5, 5, 5.5, 6, 6.5, 7, 7.5]
    
    transfer_metric = defaultdict(lambda: {'rmse': [], 'rmse_std': [], 'success': []})
    for method in data_paths.keys():
        transfer_metric.update({method: {'rmse': [], 'rmse_std': []}})
        for T in episode_len_list:
            rmse, rmse_std = [], []
            for seed in seeds:
                path = data_paths[method] + "/seed"+ str(seed) +"_*/transfer_metric_"+str(T)+".npy"
                file_path = glob.glob(path)
                if file_path:
                    temp = np.load(file_path[0], allow_pickle=True).item()
                    rmse.append(temp['rmse'])
                    rmse_std.append(temp['rmse_std'])
            if rmse:
                transfer_metric[method]['rmse'].append(np.array(rmse).mean())
                transfer_metric[method]['rmse_std'].append(np.array(rmse_std).mean())
                log(f"Performance of {method} for episode length of {T} is {np.array(rmse).mean():.6f} +/- {np.array(rmse_std).mean():.6f}")
        transfer_metric[method]['rmse'] = np.array(transfer_metric[method]['rmse'])
        transfer_metric[method]['rmse_std'] = np.array(transfer_metric[method]['rmse_std'])
        log("")

    # Create generalization plot
    fig = plt.figure(figsize=(8, 2))
    for method in data_paths.keys():
        if len(transfer_metric[method]['rmse']) > 0:
            plt.plot(episode_len_list2, transfer_metric[method]['rmse'], label=method, color=plot_colors[method])
            plt.fill_between(episode_len_list2, 
                             transfer_metric[method]['rmse']-transfer_metric[method]['rmse_std'],  
                             transfer_metric[method]['rmse']+transfer_metric[method]['rmse_std'], color=plot_colors[method], alpha=0.1)

    plt.axvline(x=5.5, linestyle='-.', color=plot_colors['iLQR'], label= 'Nominal task')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,0.2)
    plt.gca().invert_xaxis()
    plt.xlabel("figure 8 trajectory period (s)")
    plt.ylabel("rmse")
    plt.savefig(plot_dir / "generalization_curve.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(plot_dir / "generalization_curve.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Save generalization data
    np.save(data_dir / 'generalization_results_generalization.npy', dict(transfer_metric))

    # Save transfer metric data in the format expected by load_metric function
    method_to_ctrl_mapping = {
        "PPO": "ppo",
        "SAC": "sac", 
        "DPPO": "dppo",
        "PPO-MPC": "ppo_mpc"
    }

    for method in data_paths.keys():
        if method in method_to_ctrl_mapping:
            ctrl_name = method_to_ctrl_mapping[method]
            
            results = {}
            
            # Add transfer metric data for each episode length
            for i, T in enumerate(episode_len_list):
                episode_key = f'_{T}'
                if i < len(transfer_metric[method]['rmse']):
                    results[episode_key] = {
                        'mean_rmse': transfer_metric[method]['rmse'][i],
                        'std_rmse': transfer_metric[method]['rmse_std'][i]
                    }
            
            # Add inference time
            if method in inference_times:
                results['inference_time'] = {
                    'mean': inference_times[method]['mean'],
                    'std': inference_times[method]['std']
                }
            else:
                results['inference_time'] = {
                    'mean': 0.0,
                    'std': 0.0
                }
            
            # Save the results file
            filename = data_dir / f"{ctrl_name}_gen_results.npy"
            np.save(filename, results)
            log(f"Saved {method} transfer metric data to {filename}")

    log("\n")

    # =============================================================================
    # Robustness Analysis - Observation Noise
    # =============================================================================
    log("=== ROBUSTNESS ANALYSIS: OBSERVATION NOISE ===")
    
    metric = defaultdict(lambda: defaultdict(lambda: {'rmse': [], 'rmse_std': [], 'success': []}))
    noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
                  110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    
    for method in data_paths.keys():
        print(method)
        for seed in seeds:
            for ns in noise_scale:
                path = data_paths[method] + "/seed"+ str(seed) +"_*/robust_metric_ob_"+str(ns)+".npy"
                file_path = glob.glob(path)
                if file_path:
                    temp = np.load(file_path[0], allow_pickle=True).item()
                    metric[method][ns]['rmse'].append(temp['rmse'])
                    metric[method][ns]['rmse_std'].append(temp['rmse_std'])

    # Create observation noise robustness plots
    threshold = 200
    fig = plt.figure(figsize=(8, 2))
    plt.plot(noise_scale, [threshold]*len(noise_scale), color='grey', linestyle='-.', label='Relative Perf='+str(threshold)+'%')
    
    for method in metric.keys():
        temp, temp_std1, temp_std2 = [], [], []
        traj_success_till, perf0 = 200, None
        for k, ns in enumerate(noise_scale):
            if metric[method][ns]['rmse']:
                data = np.array(metric[method][ns]['rmse'])
                temp.append(data.mean())
                if perf0 is None:
                    perf0 = data.mean()
                temp_std1.append(data.mean() - data.std())
                temp_std2.append(data.mean() + data.std())
                if data.mean()*100/perf0 > threshold and traj_success_till > ns:
                    traj_success_till = ns
        
        if temp:
            temp = np.array(temp)
            temp_std1, temp_std2 = np.array(temp_std1), np.array(temp_std2)
            log(f"Observation noise robustness for {method} is {traj_success_till}")
            
            valid_indices = [i for i, ns in enumerate(noise_scale) if i < len(temp)]
            valid_noise_scale = [noise_scale[i] for i in valid_indices]
            
            plt.plot(valid_noise_scale, temp*100/temp[0], color=plot_colors[method], label=method)
            plt.fill_between(valid_noise_scale, temp_std1*100/temp[0], temp_std2*100/temp[0], color=plot_colors[method], alpha=0.1)
            plt.axvline(x=traj_success_till, linestyle='-.', color=plot_colors[method])

    plt.legend()
    plt.ylim(0,3000)
    plt.xlim(0,100)
    plt.xlabel("Noise Scale")
    plt.ylabel("Relative Performance %")
    plt.savefig(robustness_dir / "robustness_ob_rl_relative.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(robustness_dir / "robustness_ob_rl_relative.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Absolute performance plot
    threshold = 0.2
    fig = plt.figure(figsize=(8, 2))
    plt.plot(noise_scale, [threshold]*len(noise_scale), color='grey', linestyle='-.', label='rmse='+str(threshold))
    
    for method in metric.keys():
        temp, temp_std1, temp_std2 = [], [], []
        traj_success_till = 200
        for k, ns in enumerate(noise_scale):
            if metric[method][ns]['rmse']:
                data = np.array(metric[method][ns]['rmse'])
                temp.append(data.mean())
                temp_std1.append(data.mean() - data.std())
                temp_std2.append(data.mean() + data.std())
                if data.mean() > threshold and traj_success_till > ns:
                    traj_success_till = ns
        
        if temp:
            temp = np.array(temp)
            temp_std1, temp_std2 = np.array(temp_std1), np.array(temp_std2)
            log(f"Observation noise robustness (absolute) for {method} is {traj_success_till}")
            
            valid_indices = [i for i, ns in enumerate(noise_scale) if i < len(temp)]
            valid_noise_scale = [noise_scale[i] for i in valid_indices]
            
            plt.plot(valid_noise_scale, temp, color=plot_colors[method], label=method)
            plt.fill_between(valid_noise_scale, temp_std1, temp_std2, color=plot_colors[method], alpha=0.1)
            plt.axvline(x=traj_success_till, linestyle='-.', color=plot_colors[method])

    plt.legend()
    plt.ylim(0,1.0)
    plt.xlim(0,100)
    plt.xlabel("Noise Scale")
    plt.ylabel("rmse")
    plt.savefig(robustness_dir / "robustness_ob_rl_absolute.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(robustness_dir / "robustness_ob_rl_absolute.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Save observation noise data
    obs_noise_data = {}
    for method in metric.keys():
        obs_noise_data[method] = {
            'noise_factor': noise_scale,
            'rmse_mean': [],
            'rmse_std': [],
            'rmse_degradation_mean': [],
            'rmse_degradation_std': []
        }
        rmse_baseline = None  # Will store RMSE at noise level 0
        
        for ns in noise_scale:
            if metric[method][ns]['rmse']:
                data = np.array(metric[method][ns]['rmse'])
                rmse_mean = data.mean()
                rmse_std = data.std()
                obs_noise_data[method]['rmse_mean'].append(rmse_mean)
                obs_noise_data[method]['rmse_std'].append(rmse_std)
                
                # Set baseline (noise level 0) for degradation calculation
                if ns == 0:
                    rmse_baseline = rmse_mean
                
                # Calculate degradation percentage
                if rmse_baseline is not None and rmse_baseline > 0:
                    degradation_mean = (rmse_mean / rmse_baseline) * 100
                    degradation_std = (rmse_std / rmse_baseline) * 100
                else:
                    degradation_mean = 100  # Default to 100% if no baseline
                    degradation_std = 0
                
                obs_noise_data[method]['rmse_degradation_mean'].append(degradation_mean)
                obs_noise_data[method]['rmse_degradation_std'].append(degradation_std)
            else:
                obs_noise_data[method]['rmse_mean'].append(np.nan)
                obs_noise_data[method]['rmse_std'].append(np.nan)
                obs_noise_data[method]['rmse_degradation_mean'].append(np.nan)
                obs_noise_data[method]['rmse_degradation_std'].append(np.nan)
        
        obs_noise_data[method]['rmse_mean'] = np.array(obs_noise_data[method]['rmse_mean'])
        obs_noise_data[method]['rmse_std'] = np.array(obs_noise_data[method]['rmse_std'])
        obs_noise_data[method]['rmse_degradation_mean'] = np.array(obs_noise_data[method]['rmse_degradation_mean'])
        obs_noise_data[method]['rmse_degradation_std'] = np.array(obs_noise_data[method]['rmse_degradation_std'])

    for method, ctrl_name in method_to_ctrl_mapping.items():
        if method in obs_noise_data:
            np.save(data_dir / f"{ctrl_name}_obs_noise_results.npy", obs_noise_data[method])

    log("\n")

    # =============================================================================
    # Robustness Analysis - Process Noise
    # =============================================================================
    log("=== ROBUSTNESS ANALYSIS: PROCESS NOISE ===")
    
    metric = defaultdict(lambda: defaultdict(lambda: {'rmse': [], 'rmse_std': [], 'success': []}))
    noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    
    for method in data_paths.keys():
        print(method)
        for seed in seeds:
            for ns in noise_scale:
                path = data_paths[method] + "/seed"+ str(seed) +"_*/robust_metric_ps_"+str(ns)+".npy"
                file_path = glob.glob(path)
                if file_path:
                    temp = np.load(file_path[0], allow_pickle=True).item()
                    metric[method][ns]['rmse'].append(temp['rmse'])
                    metric[method][ns]['rmse_std'].append(temp['rmse_std'])

    # Create process noise robustness plots (similar structure as observation noise)
    threshold = 200
    fig = plt.figure(figsize=(8, 2))
    plt.plot(noise_scale, [threshold]*len(noise_scale), color='grey', linestyle='-.', label='Relative perf='+str(threshold)+'%')
    
    for method in metric.keys():
        temp, temp_std1, temp_std2 = [], [], []
        traj_success_till, perf0 = 200, None
        for k, ns in enumerate(noise_scale):
            if metric[method][ns]['rmse']:
                data = np.array(metric[method][ns]['rmse'])
                temp.append(data.mean())
                if perf0 is None:
                    perf0 = data.mean()
                temp_std1.append(data.mean() - data.std())
                temp_std2.append(data.mean() + data.std())
                if data.mean()*100/perf0 > threshold and traj_success_till > ns:
                    traj_success_till = ns
        
        if temp:
            temp = np.array(temp)
            temp_std1, temp_std2 = np.array(temp_std1), np.array(temp_std2)
            log(f"Process noise robustness for {method} is {traj_success_till}")
            
            valid_indices = [i for i, ns in enumerate(noise_scale) if i < len(temp)]
            valid_noise_scale = [noise_scale[i] for i in valid_indices]
            
            plt.plot(valid_noise_scale, temp*100/temp[0], color=plot_colors[method], label=method)
            plt.fill_between(valid_noise_scale, temp_std1*100/temp[0], temp_std2*100/temp[0], color=plot_colors[method], alpha=0.1)
            plt.axvline(x=traj_success_till, linestyle='-.', color=plot_colors[method])

    plt.legend()
    plt.ylim(0,3000)
    plt.xlim(0,100)
    plt.xlabel("Noise Scale")
    plt.ylabel("Relative Performance %")
    plt.savefig(robustness_dir / "robustness_ps_rl_relative.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(robustness_dir / "robustness_ps_rl_relative.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Save process noise data
    proc_noise_data = {}
    for method in metric.keys():
        proc_noise_data[method] = {
            'noise_factor': noise_scale,
            'rmse_mean': [],
            'rmse_std': [],
            'rmse_degradation_mean': [],
            'rmse_degradation_std': []
        }
        rmse_baseline = None  # Will store RMSE at noise level 0
        
        for ns in noise_scale:
            if metric[method][ns]['rmse']:
                data = np.array(metric[method][ns]['rmse'])
                rmse_mean = data.mean()
                rmse_std = data.std()
                proc_noise_data[method]['rmse_mean'].append(rmse_mean)
                proc_noise_data[method]['rmse_std'].append(rmse_std)
                
                # Set baseline (noise level 0) for degradation calculation
                if ns == 0:
                    rmse_baseline = rmse_mean
                
                # Calculate degradation percentage
                if rmse_baseline is not None and rmse_baseline > 0:
                    degradation_mean = (rmse_mean / rmse_baseline) * 100
                    degradation_std = (rmse_std / rmse_baseline) * 100
                else:
                    degradation_mean = 100  # Default to 100% if no baseline
                    degradation_std = 0
                
                proc_noise_data[method]['rmse_degradation_mean'].append(degradation_mean)
                proc_noise_data[method]['rmse_degradation_std'].append(degradation_std)
            else:
                proc_noise_data[method]['rmse_mean'].append(np.nan)
                proc_noise_data[method]['rmse_std'].append(np.nan)
                proc_noise_data[method]['rmse_degradation_mean'].append(np.nan)
                proc_noise_data[method]['rmse_degradation_std'].append(np.nan)
        
        proc_noise_data[method]['rmse_mean'] = np.array(proc_noise_data[method]['rmse_mean'])
        proc_noise_data[method]['rmse_std'] = np.array(proc_noise_data[method]['rmse_std'])
        proc_noise_data[method]['rmse_degradation_mean'] = np.array(proc_noise_data[method]['rmse_degradation_mean'])
        proc_noise_data[method]['rmse_degradation_std'] = np.array(proc_noise_data[method]['rmse_degradation_std'])

    for method, ctrl_name in method_to_ctrl_mapping.items():
        if method in proc_noise_data:
            np.save(data_dir / f"{ctrl_name}_proc_noise_results.npy", proc_noise_data[method])

    log("\n")

    # =============================================================================
    # Robustness Analysis - Parametric Noise
    # =============================================================================
    log("=== ROBUSTNESS ANALYSIS: PARAMETRIC NOISE ===")
    
    metric = defaultdict(lambda: defaultdict(lambda: {'rmse': [], 'rmse_std': [], 'success': []}))
    noise_scale = [0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    
    for method in data_paths.keys():
        print(method)
        for seed in seeds:
            for ns in noise_scale:
                path = data_paths[method] + "/seed"+ str(seed) +"_*/robust_metric_pm_"+str(ns)+".npy"
                file_path = glob.glob(path)
                if file_path:
                    temp = np.load(file_path[0], allow_pickle=True).item()
                    metric[method][ns]['rmse'].append(temp['rmse'])
                    metric[method][ns]['rmse_std'].append(temp['rmse_std'])

    # Create parametric noise robustness plots
    threshold = 200
    fig = plt.figure(figsize=(8, 2))
    plt.plot(noise_scale, [threshold]*len(noise_scale), color='grey', linestyle='-.', label='Relative perf='+str(threshold)+'%')
    
    for method in metric.keys():
        temp, tempstd, temp_std1, temp_std2 = [], [], [], []
        traj_success_till, perf0 = 200, None
        for k, ns in enumerate(noise_scale):
            if metric[method][ns]['rmse']:
                data = np.array(metric[method][ns]['rmse'])
                temp.append(data.mean())
                tempstd.append(data.std())
                temp_std1.append(data.mean() - 1*data.std())
                temp_std2.append(data.mean() + 1*data.std())
                if perf0 is None:
                    perf0 = data.mean()
                if data.mean()*100/perf0 > threshold and traj_success_till > ns:
                    traj_success_till = ns
        
        if temp:
            temp = np.array(temp)
            tempstd = np.array(tempstd)
            temp_std1, temp_std2 = np.array(temp_std1), np.array(temp_std2)
            log(f"Parametric noise robustness for {method} is {traj_success_till}")
            
            valid_indices = [i for i, ns in enumerate(noise_scale) if i < len(temp)]
            valid_noise_scale = [noise_scale[i] for i in valid_indices]
            
            plt.plot(valid_noise_scale, temp*100/temp[0], color=plot_colors[method], label=method)
            plt.fill_between(valid_noise_scale, temp_std1*100/temp[0], temp_std2*100/temp[0], color=plot_colors[method], alpha=0.1)
            plt.axvline(x=traj_success_till, linestyle='-.', color=plot_colors[method])

    plt.legend()
    plt.ylim(0,3000)
    plt.xlim(0,15)
    plt.xlabel("Noise Scale")
    plt.ylabel("Relative Performance %")
    plt.savefig(robustness_dir / "robustness_pm_rl_relative.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(robustness_dir / "robustness_pm_rl_relative.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Save parametric noise data
    param_noise_data = {}
    for method in metric.keys():
        param_noise_data[method] = {
            'noise_factor': noise_scale,
            'rmse_mean': [],
            'rmse_std': [],
            'rmse_degradation_mean': [],
            'rmse_degradation_std': []
        }
        rmse_baseline = None  # Will store RMSE at noise level 0
        
        for ns in noise_scale:
            if metric[method][ns]['rmse']:
                data = np.array(metric[method][ns]['rmse'])
                rmse_mean = data.mean()
                rmse_std = data.std()
                param_noise_data[method]['rmse_mean'].append(rmse_mean)
                param_noise_data[method]['rmse_std'].append(rmse_std)
                
                # Set baseline (noise level 0) for degradation calculation
                if ns == 0:
                    rmse_baseline = rmse_mean
                
                # Calculate degradation percentage
                if rmse_baseline is not None and rmse_baseline > 0:
                    degradation_mean = (rmse_mean / rmse_baseline) * 100
                    degradation_std = (rmse_std / rmse_baseline) * 100
                else:
                    degradation_mean = 100  # Default to 100% if no baseline
                    degradation_std = 0
                
                param_noise_data[method]['rmse_degradation_mean'].append(degradation_mean)
                param_noise_data[method]['rmse_degradation_std'].append(degradation_std)
            else:
                param_noise_data[method]['rmse_mean'].append(np.nan)
                param_noise_data[method]['rmse_std'].append(np.nan)
                param_noise_data[method]['rmse_degradation_mean'].append(np.nan)
                param_noise_data[method]['rmse_degradation_std'].append(np.nan)
        
        param_noise_data[method]['rmse_mean'] = np.array(param_noise_data[method]['rmse_mean'])
        param_noise_data[method]['rmse_std'] = np.array(param_noise_data[method]['rmse_std'])
        param_noise_data[method]['rmse_degradation_mean'] = np.array(param_noise_data[method]['rmse_degradation_mean'])
        param_noise_data[method]['rmse_degradation_std'] = np.array(param_noise_data[method]['rmse_degradation_std'])

    for method, ctrl_name in method_to_ctrl_mapping.items():
        if method in param_noise_data:
            np.save(data_dir / f"{ctrl_name}_param_results.npy", param_noise_data[method])

    log("\n")

    # =============================================================================
    # Robustness Analysis - Downwash
    # =============================================================================
    log("=== ROBUSTNESS ANALYSIS: DOWNWASH ===")
    
    metric = defaultdict(lambda: defaultdict(lambda: {'rmse': [], 'rmse_std': [], 'success': []}))
    downwash_height = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5]
    
    for method in data_paths.keys():
        print(method)
        for seed in seeds:
            for ns in downwash_height:
                path = data_paths[method] + "/seed"+ str(seed) +"_*/robust_metric_dw_"+str(ns)+".npy"
                file_path = glob.glob(path)
                if file_path:
                    temp = np.load(file_path[0], allow_pickle=True).item()
                    metric[method][ns]['rmse'].append(temp['rmse'])
                    metric[method][ns]['rmse_std'].append(temp['rmse_std'])

    # Create downwash robustness plot
    fig = plt.figure(figsize=(8, 2))

    for method in metric.keys():
        temp, temp_std1, temp_std2 = [], [], []
        for k, ns in enumerate(downwash_height):
            if metric[method][ns]['rmse']:
                data = np.array(metric[method][ns]['rmse'])
                temp.append(data.mean())
                temp_std1.append(data.mean() - data.std())
                temp_std2.append(data.mean() + data.std())
        
        if temp:
            temp = np.array(temp)
            temp_std1, temp_std2 = np.array(temp_std1), np.array(temp_std2)
            
            valid_indices = [i for i, ns in enumerate(downwash_height) if i < len(temp)]
            valid_downwash_height = [downwash_height[i] for i in valid_indices]
            
            plt.plot(valid_downwash_height, temp, color=plot_colors[method], label=method)
            plt.fill_between(valid_downwash_height, temp_std1, temp_std2, color=plot_colors[method], alpha=0.1)

    plt.plot(downwash_height, [0.1]*len(downwash_height), color='grey', linestyle='-.', label='rmse=0.1')
    plt.legend()
    plt.ylim(0,0.25)
    plt.xlabel("Downwash position (m)")
    plt.ylabel("rmse")
    plt.savefig(robustness_dir / "robustness_dw_rl.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(robustness_dir / "robustness_dw_rl.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()

    log("\n")

    # =============================================================================
    # Performance Report
    # =============================================================================
    log("=== PERFORMANCE REPORT ===")
    print("\n".join(log_messages))
    
    # Save log to file
    with open(data_dir / 'rl_analysis_report.txt', 'w') as f:
        f.write("\n".join(log_messages))
    
    log(f"Analysis complete! Results saved to:")
    log(f"  - Plots: {plot_dir}")
    log(f"  - Data: {data_dir}")
    log(f"  - Report: {data_dir / 'rl_analysis_report.txt'}")

if __name__ == "__main__":
    main()
