"""
Plot noise robustness analysis for different controller types.

Usage:
    python plot_noise_all.py <noise_type> [method_type]
    
Arguments:
    noise_type: 'obs_noise', 'proc_noise', or 'param'
    method_type: 'control', 'rl', or 'all' (default: 'all')
    
Examples:
    python plot_noise_all.py obs_noise all        # Plot all methods for observation noise
    python plot_noise_all.py proc_noise control   # Plot only control methods for process noise
    python plot_noise_all.py param rl            # Plot only RL methods for parametric noise
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from benchmarking_sim.quadrotor.benchmark_util.utils \
    import plot_colors, tag_ctrl_list

# script dir
script_dir = Path(__file__).parent.resolve()

max_seed = 10
metric_name = 'metrics.txt'
s = 2 # times std


if len(sys.argv) > 1:
    noise_option = sys.argv[1]
else:
    noise_option = 'obs_noise'
    # noise_option = 'proc_noise'
    # noise_option = 'param'

# Check for method type filter (control, rl, or all)
if len(sys.argv) > 2:
    method_type = sys.argv[2].lower()
    if method_type not in ['control', 'rl', 'all']:
        print(f"Warning: Invalid method type '{method_type}'. Using 'all'.")
        method_type = 'all'
else:
    method_type = 'all'

print('noise_option', noise_option)
print('method_type', method_type)

# Initialize empty noise_data dictionary
noise_data = {}

# Load model-based controller data if requested
if method_type in ['control', 'all']:
    print("Loading model-based controller data...")
    linear_mpc_data = np.load(script_dir / f'../data/linear_mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
    mpc_data = np.load(script_dir / f'../data/mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
    gpmpc_data = np.load(script_dir / f'../data/gpmpc_acados_TP_{noise_option}_results.npy', allow_pickle=True).item()
    ilqr_data = np.load(script_dir / f'../data/ilqr_{noise_option}_results.npy', allow_pickle=True).item()
    lqr_data = np.load(script_dir / f'../data/lqr_{noise_option}_results.npy', allow_pickle=True).item()
    pid_data = np.load(script_dir / f'../data/pid_{noise_option}_results.npy', allow_pickle=True).item()
    fmpc_data = np.load(script_dir / f'../data/fmpc_{noise_option}_results.npy', allow_pickle=True).item()
    
    # Add model-based controllers to noise_data
    noise_data.update({
        'Linear MPC': linear_mpc_data, 
        'Nonlinear MPC': mpc_data, 
        'GP-MPC': gpmpc_data,
        'iLQR': ilqr_data,
        'LQR': lqr_data,
        'Geometric Control': pid_data,
        'F-MPC': fmpc_data,
    })

# Load RL method data if requested
if method_type in ['rl', 'all']:
    print("Loading RL method data...")
    ppo_data = np.load(script_dir / f'../data/ppo_{noise_option}_results.npy', allow_pickle=True).item()
    sac_data = np.load(script_dir / f'../data/sac_{noise_option}_results.npy', allow_pickle=True).item()
    dppo_data = np.load(script_dir / f'../data/dppo_{noise_option}_results.npy', allow_pickle=True).item()
    ppo_mpc_data = np.load(script_dir / f'../data/ppo_mpc_{noise_option}_results.npy', allow_pickle=True).item()
    
    # Add RL controllers to noise_data
    noise_data.update({
        'PPO': ppo_data,
        'SAC': sac_data,
        'DPPO': dppo_data,
        'PPO-MPC': ppo_mpc_data,
    })

# Determine legend columns based on number of methods
num_methods = len(noise_data)
if num_methods <= 4:
    legend_cols = 2
elif num_methods <= 7:
    legend_cols = 3
else:
    legend_cols = 4

print(f"Loaded {num_methods} methods, using {legend_cols} legend columns")
    
# Define noise scales based on noise type (use the model-based controller scales for consistency)
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

# Function to interpolate or align data to common noise scale
def align_data_to_scale(data, target_scale):
    """Align data to target noise scale by interpolation or subsetting."""
    original_scale = data['noise_factor']
    original_rmse_deg_mean = data['rmse_degradation_mean']
    original_rmse_deg_std = data['rmse_degradation_std']
    original_rmse_mean = data.get('rmse_mean', None)
    original_rmse_std = data.get('rmse_std', None)
    
    # Find common indices
    aligned_rmse_deg_mean = []
    aligned_rmse_deg_std = []
    aligned_rmse_mean = [] if original_rmse_mean is not None else None
    aligned_rmse_std = [] if original_rmse_std is not None else None
    
    for target_val in target_scale:
        # Find closest match in original scale
        if target_val in original_scale:
            idx = list(original_scale).index(target_val)
            aligned_rmse_deg_mean.append(original_rmse_deg_mean[idx])
            aligned_rmse_deg_std.append(original_rmse_deg_std[idx])
            if original_rmse_mean is not None:
                aligned_rmse_mean.append(original_rmse_mean[idx])
                aligned_rmse_std.append(original_rmse_std[idx])
        else:
            # Use interpolation or NaN for missing values
            aligned_rmse_deg_mean.append(np.nan)
            aligned_rmse_deg_std.append(np.nan)
            if original_rmse_mean is not None:
                aligned_rmse_mean.append(np.nan)
                aligned_rmse_std.append(np.nan)
    
    aligned_data = data.copy()
    aligned_data['rmse_degradation_mean'] = np.array(aligned_rmse_deg_mean)
    aligned_data['rmse_degradation_std'] = np.array(aligned_rmse_deg_std)
    if original_rmse_mean is not None:
        aligned_data['rmse_mean'] = np.array(aligned_rmse_mean)
        aligned_data['rmse_std'] = np.array(aligned_rmse_std)
    
    return aligned_data

# Align all data to common noise scale
for method in noise_data.keys():
    noise_data[method] = align_data_to_scale(noise_data[method], noise_scale)

fig = plt.figure(figsize=(8, 3))

# plot rmse degradation for all methods
double_results = {}

for method in noise_data.keys():
    print(method)
    # Get aligned data and filter out NaN values
    x_data = np.array(noise_scale)
    y_mean = noise_data[method]['rmse_degradation_mean']
    y_std = noise_data[method]['rmse_degradation_std']
    
    # Filter out NaN values
    valid_mask = ~np.isnan(y_mean)
    if np.any(valid_mask):
        x_valid = x_data[valid_mask]
        y_mean_valid = y_mean[valid_mask]
        y_std_valid = y_std[valid_mask]
        
        plt.plot(x_valid, y_mean_valid, label=method, color=plot_colors[method])
        plt.fill_between(x_valid, 
                         y_mean_valid - s * y_std_valid,  
                         y_mean_valid + s * y_std_valid, color=plot_colors[method], alpha=0.1)
        
        # Find when RMSE degradation exceeds 200% using valid data
        for i, scale in enumerate(x_valid):
            if y_mean_valid[i] > 200:
                print(f"Method {method} reaches 200% performance at noise scale {scale}")
                double_results[method] = scale
                break
    
plt.xlabel("Noise Scale")
plt.ylabel("Relative Performance %")
if noise_option == 'obs_noise':
    plt.legend(ncol=legend_cols, fontsize=9)
    plt.ylim(0, 1200)
    plt.xlim(0, 100)
    plt.title("Performance Degradation with Observation Noise")
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'proc_noise':
    plt.legend(ncol=legend_cols, loc='upper right', fontsize=9)
    plt.title("Performance Degradation with Process Noise")
    plt.plot(noise_scale, [200]*len(noise_scale), \
             color='grey', linestyle='-.', label='Relative Perf=200%')
    # plot vertical line at 200% performance
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'param':
    plt.title("Performance Degradation with Parametric Uncertainty")
    plt.xlabel("Randomization scale")
    plt.ylim(0, 2000)
    plt.xlim(0, 5)
    plt.legend(ncol=legend_cols, loc='upper right', fontsize=9)
    for method in double_results.keys():
        plt.axvline(x=double_results[method], linestyle='--', color=plot_colors[method])
# plot an empty line for the legend
plt.plot(0, np.nan, linestyle='--', color='grey', label='Relative Perf=200%')

plt.plot(noise_scale, [200]*len(noise_scale), \
            color='grey', linestyle='-.', label='Relative Perf=200%')
plot_save_name = f"robustness_{method_type}_methods_{noise_option}"
plot_save_path = script_dir / 'noise' / f'{plot_save_name}.png'
plt.savefig(plot_save_path,bbox_inches="tight", pad_inches=0.1)
plt.savefig(plot_save_path.with_suffix('.pdf'),bbox_inches="tight", pad_inches=0.1)
print(f"plots saved as {plot_save_path}")

# Absolute performance degradation
fig_abs = plt.figure(figsize=(8, 3))

# abs_rmse_threshold = 0.1
abs_rmse_threshold = 0.25
abs_rmse_threshold_results = {}

for method in noise_data.keys():
    if 'rmse_mean' in noise_data[method] and 'rmse_std' in noise_data[method]:
        # Get aligned data and filter out NaN values
        x_data = np.array(noise_scale)
        y_mean = noise_data[method]['rmse_mean']
        y_std = noise_data[method]['rmse_std']
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y_mean)
        if np.any(valid_mask):
            x_valid = x_data[valid_mask]
            y_mean_valid = y_mean[valid_mask]
            y_std_valid = y_std[valid_mask]
            
            plt.plot(x_valid, y_mean_valid, label=method, color=plot_colors[method])
            plt.fill_between(x_valid,
                             y_mean_valid - s * y_std_valid,
                             y_mean_valid + s * y_std_valid,
                             color=plot_colors[method], alpha=0.1)
            
            # Check when absolute rmse exceeds threshold using valid data
            for i, scale in enumerate(x_valid):
                if y_mean_valid[i] > abs_rmse_threshold:
                    print(f"Method {method} exceeds absolute RMSE of {abs_rmse_threshold} at noise scale {scale}")
                    abs_rmse_threshold_results[method] = scale
                    break
    else:
        print(f"Warning: 'rmse_mean' or 'rmse_std' not found for method {method}. Skipping absolute plot.")

plt.xlabel("Noise Scale")
plt.ylabel("RMSE [m]")
if noise_option == 'obs_noise':
    plt.legend(ncol=legend_cols, fontsize=9)
    plt.xlim(0, 100)
    plt.ylim(0, 0.3)
    plt.title("Absolute Performance with Observation Noise")
    for method in abs_rmse_threshold_results.keys():
        plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'proc_noise':
    plt.legend(ncol=legend_cols, loc='upper left', fontsize=9)
    plt.ylim(0, 1.5)
    plt.title("Absolute Performance with Process Noise")
    for method in abs_rmse_threshold_results.keys():
        plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])
elif noise_option == 'param':
    plt.title("Absolute Performance with Parametric Uncertainty")
    plt.xlabel("Randomization scale")
    plt.xlim(0, 5)
    plt.ylim(0, 0.5)
    plt.legend(ncol=legend_cols, loc='upper left', fontsize=9)
    for method in abs_rmse_threshold_results.keys():
        plt.axvline(x=abs_rmse_threshold_results[method], linestyle='--', color=plot_colors[method])

plt.plot(noise_scale, [abs_rmse_threshold]*len(noise_scale), \
            color='grey', linestyle='-.', label=f'RMSE={abs_rmse_threshold}')

plot_save_name_abs = f"robustness_{method_type}_methods_{noise_option}_absolute"
plot_save_path_abs = script_dir / 'noise' / f'{plot_save_name_abs}.png'
plt.savefig(plot_save_path_abs, bbox_inches="tight", pad_inches=0.1)
plt.savefig(plot_save_path_abs.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
print(f"plots saved as {plot_save_path_abs}")

def create_ridge_plot():
    """Create a ridge plot showing RMSE distributions for each controller using only actual data."""
    
    # Calculate actual data bounds for each controller first
    controller_bounds = {}
    all_rmse_values = []
    
    for method in noise_data.keys():
        if 'rmse_mean' in noise_data[method]:
            # Get only actual RMSE values (no synthetic data)
            rmse_values = noise_data[method]['rmse_mean']
            valid_rmse = [val for val in rmse_values if not (np.isnan(val) or np.isinf(val))]
            
            if valid_rmse:
                controller_bounds[method] = {
                    'min': min(valid_rmse),
                    'max': max(valid_rmse),
                    'values': valid_rmse
                }
                all_rmse_values.extend(valid_rmse)
    
    if not all_rmse_values:
        print("No valid data for ridge plot")
        return None
    
    # Calculate global RMSE range from actual data only
    global_rmse_min = max(0, min(all_rmse_values) * 0.95)  # Small padding
    global_rmse_max = max(all_rmse_values) * 1.05  # Small padding
    
    print(f"Global RMSE range: [{global_rmse_min:.4f}, {global_rmse_max:.4f}]")
    
    # Create ridge plot
    fig, axes = plt.subplots(len(controller_bounds), 1, figsize=(8, len(controller_bounds) * 1.5), 
                            sharex=True, gridspec_kw={'hspace': 0.1})
    
    if len(controller_bounds) == 1:
        axes = [axes]
    
    for i, (method, bounds) in enumerate(controller_bounds.items()):
        ax = axes[i]
        
        # Add shaded gray areas where this controller has no data
        # Left side: from global min to controller min
        if bounds['min'] > global_rmse_min:
            ax.axvspan(global_rmse_min, bounds['min'], alpha=0.2, color='gray', zorder=0)
        
        # Right side: from controller max to global max  
        if bounds['max'] < global_rmse_max:
            ax.axvspan(bounds['max'], global_rmse_max, alpha=0.2, color='gray', zorder=0)
        
        # Create KDE only within the actual data bounds (no extrapolation)
        rmse_values = np.array(bounds['values'])
        
        if len(rmse_values) > 1:
            # Use scipy's gaussian_kde for better control
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(rmse_values)
            
            # Evaluate KDE only within actual data range (no extrapolation)
            x_range = np.linspace(bounds['min'], bounds['max'], 200)
            density = kde(x_range)
            
            # Create filled area plot constrained to actual data bounds
            ax.fill_between(x_range, 0, density, 
                           color=plot_colors.get(method, 'blue'), alpha=0.7, zorder=2)
            ax.plot(x_range, density, 
                   color=plot_colors.get(method, 'blue'), linewidth=1.5, zorder=3)
            
        elif len(rmse_values) == 1:
            # Single point - show as a vertical line
            ax.axvline(x=rmse_values[0], color=plot_colors.get(method, 'blue'), 
                      linewidth=3, alpha=0.8, zorder=2)
        
        # Add mean line
        # mean_rmse = np.mean(rmse_values)
        # ax.axvline(x=mean_rmse, color='red', linestyle='--', linewidth=1, alpha=0.8, zorder=4)
        
        # Customize this subplot
        ax.set_ylim(0, None)
        ax.set_xlim(global_rmse_min, global_rmse_max)
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add controller name
        ax.text(0.02, 0.8, method, transform=ax.transAxes, 
               fontsize=12, ha='left', va='top', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # # Add statistics
        # ax.text(0.98, 0.8, f'μ={mean_rmse:.3f}\nσ={np.std(rmse_values):.3f}', 
        #        transform=ax.transAxes, fontsize=10, ha='right', va='top',
        #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # print(f"Controller {method}: RMSE range [{bounds['min']:.3f}, {bounds['max']:.3f}], mean={mean_rmse:.3f}")
    
    # Configure the bottom axis
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel('RMSE [m]', fontsize=12)
    
    # Set title
    if noise_option == 'obs_noise':
        title = 'RMSE Distribution Across Observation Noise Levels'
    elif noise_option == 'proc_noise':
        title = 'RMSE Distribution Across Process Noise Levels'
    elif noise_option == 'param':
        title = 'RMSE Distribution Across Parametric Uncertainty Levels'
    else:
        title = f'RMSE Distribution Across {noise_option} Levels'
    
    fig.suptitle(title, fontsize=14, y=0.95)
    
    # Save ridge plot
    ridge_save_name = f"ridge_plot_rmse_{method_type}_{noise_option}"
    ridge_save_path = script_dir / 'noise' / f'{ridge_save_name}.png'
    plt.savefig(ridge_save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.savefig(ridge_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
    print(f"Ridge plot saved as {ridge_save_path}")
    plt.close()
    
    return controller_bounds

# # Create an alternative simplified ridge plot 
# def create_simple_ridge_plot():
#     """Create a simpler ridge plot using violin plots."""
#     # Calculate global RMSE range for consistent axis limits
#     all_rmse_values = []
#     for method in noise_data.keys():
#         if 'rmse_mean' in noise_data[method]:
#             all_rmse_values.extend(noise_data[method]['rmse_mean'])
    
#     if all_rmse_values:
#         global_rmse_min = max(0, min(all_rmse_values))  # Ensure minimum is 0
#         global_rmse_max = max(all_rmse_values) * 1.1
#     else:
#         global_rmse_min, global_rmse_max = 0, 1
    
#     fig, axes = plt.subplots(len(noise_data), 1, figsize=(10, len(noise_data) * 1.0), 
#                             sharex=True, gridspec_kw={'hspace': 0.05})
    
#     if len(noise_data) == 1:
#         axes = [axes]
    
#     controllers = list(noise_data.keys())
    
#     for i, (method, ax) in enumerate(zip(controllers, axes)):
#         if 'rmse_mean' in noise_data[method]:
#             rmse_values = noise_data[method]['rmse_mean']
            
#             # Filter out any NaN or infinite values
#             rmse_values = np.array(rmse_values)
#             rmse_values = rmse_values[np.isfinite(rmse_values)]
            
#             if len(rmse_values) > 0:
#                 # Get minimum RMSE value for this controller to define gray area
#                 min_rmse = np.min(rmse_values)
                
#                 # Add shaded gray area where controller has no density (left of minimum)
#                 ax.axvspan(0, min_rmse, alpha=0.2, color='gray', zorder=0)
                
#                 # Create a simple histogram-style plot
#                 ax.hist(rmse_values, bins=20, alpha=0.7, color=plot_colors[method], 
#                        density=True, orientation='horizontal')
                
#                 # Add mean line
#                 mean_rmse = np.mean(rmse_values)
#                 ax.axhline(mean_rmse, color='red', linestyle='--', linewidth=2, alpha=0.8)
                
#                 # Customize appearance
#                 ax.set_xlim(0, None)  # Ensure density starts from 0
#                 ax.set_ylim(global_rmse_min, global_rmse_max)  # Use global RMSE range for y-axis
#                 ax.set_ylabel('')
#                 ax.set_xticks([])
#                 ax.spines['bottom'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.spines['top'].set_visible(False)
                
#                 # Add controller name
#                 ax.text(0.02, 0.8, method, transform=ax.transAxes, 
#                        fontsize=11, ha='left', va='top', weight='bold')
                
#                 # Add statistics
#                 ax.text(0.98, 0.8, f'μ={mean_rmse:.3f}', transform=ax.transAxes,
#                        fontsize=9, ha='right', va='top',
#                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
#                 print(f"Controller {method}: RMSE range [{np.min(rmse_values):.3f}, {np.max(rmse_values):.3f}], mean={mean_rmse:.3f}")
#             else:
#                 print(f"No valid RMSE data for {method}")
    
#     # Configure the bottom axis
#     axes[-1].spines['left'].set_visible(True)
#     axes[-1].set_ylabel('RMSE [m]', fontsize=12)
    
#     # Set title
#     if noise_option == 'obs_noise':
#         title = 'RMSE Distribution: Observation Noise'
#     elif noise_option == 'proc_noise':
#         title = 'RMSE Distribution: Process Noise'
#     elif noise_option == 'param':
#         title = 'RMSE Distribution: Parametric Uncertainty'
#     else:
#         title = f'RMSE Distribution: {noise_option}'
    
#     fig.suptitle(title, fontsize=14, y=0.95)
    
#     # Save simple ridge plot
#     simple_ridge_save_name = f"simple_ridge_plot_rmse_{noise_option}"
#     simple_ridge_save_path = script_dir / 'noise' / f'{simple_ridge_save_name}.png'
#     plt.savefig(simple_ridge_save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
#     plt.savefig(simple_ridge_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
#     print(f"Simple ridge plot saved as {simple_ridge_save_path}")
#     plt.close()

# Create ridge plots
print("Creating ridge plots...")
controller_bounds = create_ridge_plot()
# create_simple_ridge_plot()

# # Create an alternative ridge plot with better spacing and violin-like appearance  
# def create_violin_ridge_plot():
#     """Create a violin-style ridge plot."""
#     # Calculate global RMSE range for consistent axis limits
#     all_rmse_values = []
#     for method in noise_data.keys():
#         if 'rmse_mean' in noise_data[method]:
#             all_rmse_values.extend(noise_data[method]['rmse_mean'])
    
#     if all_rmse_values:
#         global_rmse_min = max(0, min(all_rmse_values))  # Ensure minimum is 0
#         global_rmse_max = max(all_rmse_values) * 1.1
#     else:
#         global_rmse_min, global_rmse_max = 0, 1
    
#     fig, axes = plt.subplots(len(noise_data), 1, figsize=(10, len(noise_data) * 1.2), 
#                             sharex=True, gridspec_kw={'hspace': 0})
    
#     if len(noise_data) == 1:
#         axes = [axes]
    
#     controllers = list(noise_data.keys())
    
#     for i, (method, ax) in enumerate(zip(controllers, axes)):
#         if 'rmse_mean' in noise_data[method]:
#             rmse_values = noise_data[method]['rmse_mean']
            
#             # Get minimum RMSE value for this controller to define gray area
#             min_rmse = np.min(rmse_values)
            
#             # Add shaded gray area where controller has no density (left of minimum)
#             ax.axvspan(global_rmse_min, min_rmse, alpha=0.2, color='gray', zorder=0)
            
#             # Create violin plot for this controller
#             parts = ax.violinplot([rmse_values], positions=[0], widths=0.8, 
#                                 showmeans=False, showmedians=True, showextrema=False)
            
#             # Customize violin appearance
#             for pc in parts['bodies']:
#                 pc.set_facecolor(plot_colors[method])
#                 pc.set_alpha(0.7)
#                 pc.set_edgecolor('black')
#                 pc.set_linewidth(1)
            
#             # Customize median line
#             if 'cmedians' in parts:
#                 parts['cmedians'].set_color('black')
#                 parts['cmedians'].set_linewidth(2)
            
#             # Set y-axis properties
#             ax.set_ylim(-0.5, 0.5)
#             ax.set_xlim(global_rmse_min, global_rmse_max)  # Use global RMSE range
#             ax.set_yticks([])
#             ax.spines['left'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.spines['top'].set_visible(False)
            
#             # Add controller name
#             ax.text(-0.15, 0, method, transform=ax.transData, 
#                    fontsize=11, ha='right', va='center', weight='bold')
            
#             # Add mean RMSE value as text
#             mean_rmse = np.mean(rmse_values)
#             ax.text(0.95, 0, f'{mean_rmse:.3f}', transform=ax.transAxes,
#                    fontsize=10, ha='right', va='center', 
#                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
#     # Configure the bottom axis
#     axes[-1].spines['bottom'].set_visible(True)
#     axes[-1].set_xlabel('RMSE [m]', fontsize=12)
    
#     # Set title
#     if noise_option == 'obs_noise':
#         title = 'RMSE Distribution: Observation Noise'
#     elif noise_option == 'proc_noise':
#         title = 'RMSE Distribution: Process Noise'
#     elif noise_option == 'param':
#         title = 'RMSE Distribution: Parametric Uncertainty'
#     else:
#         title = f'RMSE Distribution: {noise_option}'
    
#     fig.suptitle(title, fontsize=14, y=0.95)
    
#     # Save violin ridge plot
#     violin_ridge_save_name = f"violin_ridge_plot_rmse_{noise_option}"
#     violin_ridge_save_path = script_dir / 'noise' / f'{violin_ridge_save_name}.png'
#     plt.savefig(violin_ridge_save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
#     plt.savefig(violin_ridge_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
#     print(f"Violin ridge plot saved as {violin_ridge_save_path}")
#     plt.close()

# # Create the violin-style ridge plot
# create_violin_ridge_plot()