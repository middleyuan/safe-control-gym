import os
import sys
from pathlib import Path
import glob

import numpy as np
import matplotlib.pyplot as plt

from benchmarking_sim.quadrotor.benchmark_util.utils import plot_colors, plotting_data_dir

# script dir
script_dir = Path(__file__).parent.resolve()
data_dir = plotting_data_dir(script_dir)
print(script_dir)
# if the output path does not exist, create it
output_path = script_dir / 'noise'
output_path.mkdir(exist_ok=True)

def get_key_by_value(d, value):
    """Return the first key in dict d whose value matches the given value."""
    for k, v in d.items():
        if v == value:
            return k
    return None

max_seed = 10
s = 2  # times std

# RL method mapping
rl_methods = ['ppo', 'sac', 'dppo', 'ppo_mpc']
rl_method_names = {
    'ppo': 'PPO',
    'sac': 'SAC', 
    'dppo': 'DPPO',
    'ppo_mpc': 'PPO-MPC'
}

if len(sys.argv) > 1:
    controller = sys.argv[1]
    noise_type = sys.argv[2]
else:
    controller = 'ppo'  # default
    noise_type = 'obs_noise'  # default
    # noise_type = 'proc_noise'
    # noise_type = 'param'

print(f'INFO: controller: {controller}, noise_type: {noise_type}')

assert noise_type in ['obs_noise', 'proc_noise', 'param'], f'noise_type {noise_type} not supported'
assert controller in rl_methods, f'controller {controller} not supported'

# Map noise types to their file prefixes
noise_type_mapping = {
    'obs_noise': 'ob',
    'proc_noise': 'ps', 
    'param': 'pm'
}

SYS = 'quadrotor_2D_attitude'

# Define data directory for RL methods
if controller == 'sac':
    data_folder_dir = data_dir / 'nominal' / f'{SYS}_{controller}_data2'
else:
    data_folder_dir = data_dir / 'nominal' / f'{SYS}_{controller}_data'

print(f'Data folder: {data_folder_dir}')

# Check if data folder exists
if not data_folder_dir.exists():
    print(f'Error: Data folder {data_folder_dir} does not exist')
    sys.exit(1)

# Find seed folders
seed_folders = [f for f in data_folder_dir.iterdir() if f.is_dir() and f.name.startswith('seed')]
seed_folders = sorted(seed_folders, key=lambda x: int(x.name.split('_')[0].replace('seed', '')))

print(f'Found {len(seed_folders)} seed folders')
if len(seed_folders) < max_seed:
    max_seed = len(seed_folders)
    print(f'Adjusting max_seed to {max_seed}')

# Define noise scales based on noise type
if noise_type == 'obs_noise':
    noise_scales = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
elif noise_type == 'proc_noise':
    noise_scales = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
elif noise_type == 'param':
    noise_scales = [0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

noise_prefix = noise_type_mapping[noise_type]

results = {}

# Process each seed
for seed_idx in range(max_seed):
    results[str(seed_idx)] = {}
    rmse_list = []
    early_stop_list = []
    noise_factor_list = []
    
    seed_folder = seed_folders[seed_idx]
    print(f'Processing {seed_folder.name}')
    
    # Process each noise scale
    for noise_scale in noise_scales:
        # Construct the filename based on noise type
        if noise_type == 'param':
            filename = f'robust_metric_{noise_prefix}_{noise_scale}.npy'
        else:
            filename = f'robust_metric_{noise_prefix}_{int(noise_scale)}.npy'
        
        file_path = seed_folder / filename
        
        if file_path.exists():
            try:
                data = np.load(file_path, allow_pickle=True).item()
                
                # Extract relevant metrics
                average_rmse = data.get('average_rmse', np.nan)
                early_stop = data.get('early_stop', [0] * 10)  # default to no early stop
                noise_factor = data.get('noise_scale', noise_scale)
                
                # Handle early stop - check if any run had early stop
                early_stop_flag = any(early_stop) if isinstance(early_stop, (list, np.ndarray)) else bool(early_stop)
                
                rmse_list.append(average_rmse)
                early_stop_list.append(early_stop_flag)
                noise_factor_list.append(noise_factor)
                
            except Exception as e:
                print(f'Error loading {file_path}: {e}')
                # Append NaN values for missing data
                rmse_list.append(np.nan)
                early_stop_list.append(True)  # Assume early stop if file can't be loaded
                noise_factor_list.append(noise_scale)
        else:
            print(f'File not found: {file_path}')
            # Append NaN values for missing files
            rmse_list.append(np.nan)
            early_stop_list.append(True)  # Assume early stop if file doesn't exist
            noise_factor_list.append(noise_scale)
    
    results[str(seed_idx)]['rmse'] = rmse_list
    results[str(seed_idx)]['early_stop'] = early_stop_list
    results[str(seed_idx)]['noise_factor'] = noise_factor_list

# Filter out noise scales that have all NaN values across all seeds
valid_indices = []
for i in range(len(noise_scales)):
    has_valid_data = False
    for seed_idx in range(max_seed):
        if not np.isnan(results[str(seed_idx)]['rmse'][i]):
            has_valid_data = True
            break
    if has_valid_data:
        valid_indices.append(i)

print(f'Valid noise scale indices: {len(valid_indices)} out of {len(noise_scales)}')

# Filter results to only include valid data points
for seed_idx in range(max_seed):
    results[str(seed_idx)]['rmse'] = [results[str(seed_idx)]['rmse'][i] for i in valid_indices]
    results[str(seed_idx)]['early_stop'] = [results[str(seed_idx)]['early_stop'][i] for i in valid_indices]
    results[str(seed_idx)]['noise_factor'] = [results[str(seed_idx)]['noise_factor'][i] for i in valid_indices]

# Update noise_scales to only include valid ones
noise_scales = [noise_scales[i] for i in valid_indices]

if len(noise_scales) == 0:
    print('Error: No valid data found')
    sys.exit(1)

# Plot individual seeds
fig, ax = plt.subplots(figsize=(10, 6))
for seed_idx in range(max_seed):
    # Filter out NaN values for plotting
    rmse_vals = results[str(seed_idx)]['rmse']
    noise_vals = results[str(seed_idx)]['noise_factor']
    
    valid_mask = ~np.isnan(rmse_vals)
    rmse_vals = np.array(rmse_vals)[valid_mask]
    noise_vals = np.array(noise_vals)[valid_mask]
    
    if len(rmse_vals) > 0:
        ax.plot(noise_vals, rmse_vals, label=f'seed_{seed_idx+1}', alpha=0.7)

ax.set_xlabel('Noise scale')
ax.set_ylabel('RMSE')
ax.set_title(f'{rl_method_names[controller]} {noise_type} RMSE')
ax.legend()
fig.savefig(output_path / f'{controller}_{noise_type}_rmse_individual.png')
plt.close()

# Compute statistics
# Convert to numpy arrays, handling NaN values
rmse_data = []
early_stop_data = []
for seed_idx in range(max_seed):
    rmse_data.append(results[str(seed_idx)]['rmse'])
    early_stop_data.append(results[str(seed_idx)]['early_stop'])

rmse = np.array(rmse_data)
early_stop = np.array(early_stop_data)
noise_factor = np.array(noise_scales)

# Compute mean and std, ignoring NaN values
rmse_mean = np.nanmean(rmse, axis=0)
rmse_std = np.nanstd(rmse, axis=0)
rmse_max = np.nanmax(rmse, axis=0)

# Compute rmse degradation relative to baseline (noise_scale = 0)
baseline_rmse = rmse[:, 0:1]  # First column (noise_scale = 0)
rmse_degradation = (rmse / baseline_rmse) * 100
rmse_degradation_mean = np.nanmean(rmse_degradation, axis=0)
rmse_degradation_std = np.nanstd(rmse_degradation, axis=0)

# Handle early stop analysis
early_stop_results = np.any(early_stop, axis=0)  # True if any seed had early stop at this noise level
first_early_stop = None
early_stop_noise_factor = None
if np.any(early_stop_results):
    first_early_stop = np.where(early_stop_results)[0][0]
    early_stop_noise_factor = noise_factor[first_early_stop]

print(f'RMSE degradation mean: {rmse_degradation_mean}')
print(f'Early stop noise factor: {early_stop_noise_factor}')

# Plot RMSE
fig, ax = plt.subplots(figsize=(6, 2))
controller_name = rl_method_names[controller]
ax.plot(noise_factor, rmse_mean, 
        label='mean', color=plot_colors.get(controller_name, 'tab:blue'))
ax.fill_between(noise_factor, rmse_mean - s*rmse_std, rmse_mean + s*rmse_std, 
                alpha=0.2, label=f'{s} std', color=plot_colors.get(controller_name, 'tab:blue'))

# Add reference line
ax.axhline(y=0.1, color='gray', linestyle='--', label='RMSE = 0.1')
ax.legend(ncol=2)
ax.set_xlabel('Noise scale')
ax.set_ylabel('RMSE')
ax.set_title(f'RMSE of {controller_name}')
fig.tight_layout()
fig.savefig(output_path / f'{controller}_{noise_type}_rmse.png')
plt.close()

# Plot RMSE degradation
fig, ax = plt.subplots(figsize=(6, 2)) 
ax.plot(noise_factor, rmse_degradation_mean, 
        label='mean', color=plot_colors.get(controller_name, 'tab:blue')) 
ax.fill_between(noise_factor, rmse_degradation_mean - s*rmse_degradation_std, 
                rmse_degradation_mean + s*rmse_degradation_std, 
                alpha=0.2, label=f'{s} std', color=plot_colors.get(controller_name, 'tab:blue'))

ax.axhline(y=200, color='gray', linestyle='--', label='Relative Perf=200%')
ax.legend()
ax.set_xlabel('Noise scale')
ax.set_ylabel('RMSE degradation (%)')
ax.set_title(f'RMSE degradation of {controller_name} with {noise_type}')
fig.tight_layout()
fig.savefig(output_path / f'{controller}_{noise_type}_rmse_degradation.png')
plt.close()

# Save results in the same format as plot_noise.py
saved_results = {
    'rmse': rmse,
    'rmse_mean': rmse_mean,
    'rmse_std': rmse_std,
    'rmse_max': rmse_max,
    'rmse_degradation_mean': rmse_degradation_mean,
    'rmse_degradation_std': rmse_degradation_std,
    'noise_factor': noise_factor,
    'early_stop_noise_factor': early_stop_noise_factor,
    'early_stop': early_stop,
}

# Save results
results_file_name = data_dir / f'{controller}_{noise_type}_results.npy'
np.save(results_file_name, saved_results)
print(f'Saved results to {results_file_name}')

print(f'Processing complete for {controller_name} with {noise_type}')
print(f'Mean RMSE at baseline: {rmse_mean[0]:.6f} ± {rmse_std[0]:.6f}')
if early_stop_noise_factor is not None:
    print(f'First early stop at noise factor: {early_stop_noise_factor}')
else:
    print('No early stops detected')
