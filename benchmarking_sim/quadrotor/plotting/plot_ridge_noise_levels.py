"""
Ridge plot visualization for controller robustness at different noise levels.

This script creates ridge plots showing RMSE distributions for different controllers
at specific noise levels, allowing comparison of robustness across methods.

Usage:
    python plot_ridge_noise_levels.py <noise_type> [method_type]
    
Arguments:
    noise_type: 'obs_noise', 'proc_noise', or 'param'
    method_type: 'control', 'rl', or 'all' (default: 'all')
    
Examples:
    python plot_ridge_noise_levels.py obs_noise all        # All methods for observation noise
    python plot_ridge_noise_levels.py proc_noise control   # Only control methods for process noise
    python plot_ridge_noise_levels.py param rl            # Only RL methods for parametric noise
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import gaussian_kde

from benchmarking_sim.quadrotor.benchmark_util.utils \
    import plot_colors, tag_ctrl_list, plotting_data_dir

# script dir
script_dir = Path(__file__).parent.resolve()
data_dir = plotting_data_dir(script_dir)

max_seed = 10
metric_name = 'metrics.txt'

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
    linear_mpc_data = np.load(data_dir / f'linear_mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
    mpc_data = np.load(data_dir / f'mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
    gpmpc_data = np.load(data_dir / f'gpmpc_acados_TP_{noise_option}_results.npy', allow_pickle=True).item()
    ilqr_data = np.load(data_dir / f'ilqr_{noise_option}_results.npy', allow_pickle=True).item()
    lqr_data = np.load(data_dir / f'lqr_{noise_option}_results.npy', allow_pickle=True).item()
    pid_data = np.load(data_dir / f'pid_{noise_option}_results.npy', allow_pickle=True).item()
    fmpc_data = np.load(data_dir / f'fmpc_{noise_option}_results.npy', allow_pickle=True).item()
    
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
    ppo_data = np.load(data_dir / f'ppo_{noise_option}_results.npy', allow_pickle=True).item()
    sac_data = np.load(data_dir / f'sac_{noise_option}_results.npy', allow_pickle=True).item()
    dppo_data = np.load(data_dir / f'dppo_{noise_option}_results.npy', allow_pickle=True).item()
    ppo_mpc_data = np.load(data_dir / f'ppo_mpc_{noise_option}_results.npy', allow_pickle=True).item()
    
    # Add RL controllers to noise_data
    noise_data.update({
        'PPO': ppo_data,
        'SAC': sac_data,
        'DPPO': dppo_data,
        'PPO-MPC': ppo_mpc_data,
    })

print(f"Loaded {len(noise_data)} methods for ridge plot") 

# Define noise scales
if noise_option in ['obs_noise']:
    noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   12, 14, 16, 18, 20, 25, 30, 35, 40, 
                   45, 50, 60, 70, 80, 90, 100]
elif noise_option == 'proc_noise':
    noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   12, 14, 16, 18, 20, 25, 30, 35, 40, 
                   45, 50, 60, 70, 80, 90, 100]
elif noise_option == 'param':
    noise_scale = [0, 0.01, 0.02, 0.05, 0.1, 
                   0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 
                   1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 
                   2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0]
    noise_scale.sort()

# Function to interpolate or align data to common noise scale
def align_data_to_scale(data, target_scale):
    """Align data to target noise scale by interpolation or subsetting."""
    original_scale = data['noise_factor']
    original_rmse_mean = data.get('rmse_mean', None)
    original_rmse_std = data.get('rmse_std', None)
    
    # Find common indices
    aligned_rmse_mean = [] if original_rmse_mean is not None else None
    aligned_rmse_std = [] if original_rmse_std is not None else None
    
    if original_rmse_mean is not None:
        for target_val in target_scale:
            # Find closest match in original scale
            if target_val in original_scale:
                idx = list(original_scale).index(target_val)
                aligned_rmse_mean.append(original_rmse_mean[idx])
                aligned_rmse_std.append(original_rmse_std[idx])
            else:
                # Use NaN for missing values
                aligned_rmse_mean.append(np.nan)
                aligned_rmse_std.append(np.nan)
        
        aligned_data = data.copy()
        aligned_data['rmse_mean'] = np.array(aligned_rmse_mean)
        aligned_data['rmse_std'] = np.array(aligned_rmse_std)
    else:
        aligned_data = data.copy()
    
    return aligned_data

# Align all data to common noise scale
for method in noise_data.keys():
    noise_data[method] = align_data_to_scale(noise_data[method], noise_scale)

def create_color_gradient(base_color, n_levels, reverse=False):
    """
    Create a color gradient from light to dark (or dark to light if reverse=True)
    based on a base color from plot_colors.
    
    Args:
        base_color: Base color name or hex code
        n_levels: Number of color levels to generate
        reverse: If True, create dark to light gradient (default: light to dark)
    
    Returns:
        List of colors from light to dark (or reversed)
    """
    # Convert color name to RGB
    rgb = mcolors.to_rgb(base_color)
    
    # Create gradient where noise level 0 gets original color, higher levels get darker
    colors = []
    
    for i in range(n_levels):
        if i == 0:
            # First color (noise level 0) gets the original color
            scaled_rgb = rgb
        else:
            # Higher noise levels get progressively darker
            # Scale from 1.0 (original) down to 0.4 (darker)
            darkness_factor = 1.0 - 0.6 * i / (n_levels - 1) if n_levels > 1 else 1.0
            scaled_rgb = tuple(c * darkness_factor for c in rgb)
        
        colors.append(scaled_rgb)
    
    if reverse:
        colors = colors[::-1]
    
    return colors

def create_ridge_plot_at_noise_levels(selected_noise_indices=None, controllers_to_plot=None, create_individual_plots=False):
    """
    Create ridge plots showing RMSE spread at specific noise levels.
    Each subplot shows one controller's RMSE distribution at the selected noise levels.
    
    Args:
        selected_noise_indices: List of indices into noise_scale to plot. 
                               If None, will automatically select representative levels.
        controllers_to_plot: List of controller names to plot. If None, plot all controllers.
                            Available controllers: 'Linear MPC', 'Nonlinear MPC', 'GP-MPC', 
                            'iLQR', 'LQR', 'Geometric Control', 'F-MPC'
        create_individual_plots: If True, create separate plot files for each controller.
    """
    
    # Filter controllers if specified
    if controllers_to_plot is not None:
        # Validate controller names
        available_controllers = list(noise_data.keys())
        invalid_controllers = [c for c in controllers_to_plot if c not in available_controllers]
        if invalid_controllers:
            print(f"Warning: Invalid controller names: {invalid_controllers}")
            print(f"Available controllers: {available_controllers}")
        
        # Filter to only include valid controllers
        valid_controllers = [c for c in controllers_to_plot if c in available_controllers]
        if not valid_controllers:
            print("No valid controllers specified. Using all controllers.")
            filtered_noise_data = noise_data
        else:
            filtered_noise_data = {k: v for k, v in noise_data.items() if k in valid_controllers}
            print(f"Plotting controllers: {list(filtered_noise_data.keys())}")
    else:
        filtered_noise_data = noise_data
    
    # Auto-select representative noise levels if not specified
    if selected_noise_indices is None:
        if len(noise_scale) >= 8:
            # Select evenly spaced indices to show progression
            selected_noise_indices = [0, len(noise_scale)//4, len(noise_scale)//2, 
                                    3*len(noise_scale)//4, len(noise_scale)-1]
        else:
            selected_noise_indices = list(range(len(noise_scale)))
    
    selected_noise_levels = [noise_scale[i] for i in selected_noise_indices]
    print(f"Selected noise levels: {selected_noise_levels}")
    
    # Generate synthetic RMSE samples for each controller at each noise level
    # Since we only have mean and std, we'll create synthetic distributions
    ridge_data = []
    
    for method in filtered_noise_data.keys():
        if 'rmse_mean' in filtered_noise_data[method] and 'rmse_std' in filtered_noise_data[method]:
            rmse_means = filtered_noise_data[method]['rmse_mean']
            rmse_stds = filtered_noise_data[method]['rmse_std']
            
            for noise_idx in selected_noise_indices:
                if noise_idx < len(rmse_means):
                    mean_val = rmse_means[noise_idx]
                    std_val = rmse_stds[noise_idx]
                    noise_level = noise_scale[noise_idx]
                    
                    if not (np.isnan(mean_val) or np.isnan(std_val) or np.isinf(mean_val) or np.isinf(std_val)):
                        # Generate synthetic samples assuming normal distribution
                        n_samples = 100
                        if std_val > 0:
                            samples = np.random.normal(mean_val, std_val, n_samples)
                            # Ensure RMSE values are non-negative
                            samples = np.maximum(samples, 0.001)
                        else:
                            # If no std, just use the mean value repeated
                            samples = np.full(n_samples, max(mean_val, 0.001))
                        
                        for sample in samples:
                            ridge_data.append({
                                'Controller': method,
                                'RMSE': sample,
                                'NoiseLevel': noise_level,
                                'NoiseIndex': noise_idx
                            })
    
    if not ridge_data:
        print("No valid data for ridge plot")
        return None
    
    # Convert to DataFrame
    df_ridge = pd.DataFrame(ridge_data)
    
    # Calculate global RMSE range for consistent x-axis
    all_rmse = df_ridge['RMSE'].values
    global_rmse_min = max(0, np.min(all_rmse) * 0.95)
    global_rmse_max = np.max(all_rmse) * 1.05
    
    print(f"Global RMSE range: [{global_rmse_min:.4f}, {global_rmse_max:.4f}]")
    
    # Create subplots - one for each controller
    n_controllers = len(filtered_noise_data)
    fig, axes = plt.subplots(n_controllers, 1, figsize=(8, n_controllers * 2), 
                            sharex=True, gridspec_kw={'hspace': 0.5})
    
    if n_controllers == 1:
        axes = [axes]
    
    for i, (method, ax) in enumerate(zip(filtered_noise_data.keys(), axes)):
        controller_data = df_ridge[df_ridge['Controller'] == method]
        
        # Get base color for this controller from plot_colors
        base_color = plot_colors.get(method, 'gray')
        
        # Create color gradient for noise levels (darker = higher noise)
        noise_colors = create_color_gradient(base_color, len(selected_noise_levels), reverse=False)
        
        if len(controller_data) > 0:
            # Plot ridge for each noise level (reverse order so highest noise is at top)
            y_offset = 0
            max_density = 0
            
            for j, (noise_idx, noise_level) in enumerate(zip(reversed(selected_noise_indices), reversed(selected_noise_levels))):
                noise_data_subset = controller_data[controller_data['NoiseIndex'] == noise_idx]
                
                if len(noise_data_subset) > 0:
                    rmse_values = noise_data_subset['RMSE'].values
                    
                    # Color assignment: higher noise levels (larger j values when reversed) get darker colors
                    # Since we're iterating in reverse order, j=0 is highest noise (should be darkest)
                    # j=len-1 is lowest noise (should be lightest)
                    color_idx = len(selected_noise_levels) - 1 - j  # This gives us the correct gradient
                    current_color = noise_colors[color_idx]
                    
                    if len(rmse_values) > 1:
                        # Create KDE for this noise level
                        kde = gaussian_kde(rmse_values)
                        x_range = np.linspace(max(global_rmse_min, np.min(rmse_values) * 0.99), 
                                            min(global_rmse_max, np.max(rmse_values) * 1.01), 200)
                        density = kde(x_range)
                        
                        # Normalize density for consistent ridge height
                        density = density / np.max(density) * 0.8
                        max_density = max(max_density, np.max(density))
                        
                        # Plot the ridge
                        ax.fill_between(x_range, y_offset, y_offset + density, 
                                       color=current_color, alpha=0.7, zorder=2)
                        ax.plot(x_range, y_offset + density, 
                               color=current_color, linewidth=1.5, zorder=3)
                        
                        # Add mean line
                        mean_rmse = np.mean(rmse_values)
                        ax.axvline(x=mean_rmse, ymin=y_offset/(len(selected_noise_levels)+0.5), 
                                  ymax=(y_offset+0.8)/(len(selected_noise_levels)+0.5),
                                  color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)
                        
                        # Add noise level label (positioned in the middle)
                        label_text = f'Noise {noise_level}'
                        if noise_option == 'param':
                            label_text = f'Param {noise_level:.2f}'
                        
                        ax.text(global_rmse_min + (global_rmse_max - global_rmse_min) * 0.5, y_offset + 0.4, label_text,
                               fontsize=9, va='center', ha='center', 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=current_color, alpha=0.8))
                        
                        # Add mean value annotation (positioned on the right)
                        ax.text(global_rmse_min + (global_rmse_max - global_rmse_min) * 0.98, y_offset + 0.4, f'{mean_rmse:.3f}',
                               fontsize=8, va='center', ha='right',
                               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
                    
                    else:
                        # Single point - show as vertical line
                        ax.axvline(x=rmse_values[0], ymin=y_offset/(len(selected_noise_levels)+0.5), 
                                  ymax=(y_offset+0.8)/(len(selected_noise_levels)+0.5),
                                  color=current_color, linewidth=3, alpha=0.8, zorder=2)
                    
                    y_offset += 1
            
            # Customize subplot
            ax.set_xlim(global_rmse_min, global_rmse_max)
            ax.set_ylim(-0.2, len(selected_noise_levels) + 0.2)
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Add controller name (positioned in the middle and higher up)
            ax.text(0.5, 1.05, method, transform=ax.transAxes, 
                   fontsize=14, ha='center', va='bottom', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
            
            if i < len(axes) - 1:
                ax.spines['bottom'].set_visible(False)
    
    # Configure the bottom axis
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel('RMSE [m]', fontsize=12)
    
    # Set title
    noise_type_titles = {
        'obs_noise': 'Observation Noise',
        'proc_noise': 'Process Noise', 
        'param': 'Parameter Uncertainty'
    }
    noise_type = noise_type_titles.get(noise_option, noise_option)
    fig.suptitle(f'RMSE Distribution at Different {noise_type} Levels', 
                fontsize=16, fontweight='bold', y=1.05)
    
    # Add a note about the noise levels
    level_str = ', '.join([str(x) for x in selected_noise_levels])
    if noise_option == 'param':
        level_str = ', '.join([f'{x:.2f}' for x in selected_noise_levels])
    
    # fig.text(0.5, 0.02, f'Noise levels shown: {level_str}', 
    #         ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save the plot
    ridge_save_name = f"ridge_plot_noise_levels_{method_type}_{noise_option}"
    ridge_save_path = script_dir / 'noise' / f'{ridge_save_name}.png'
    ridge_save_path.parent.mkdir(exist_ok=True)
    plt.savefig(ridge_save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.savefig(ridge_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
    print(f"Ridge plot saved as {ridge_save_path}")
    plt.show()
    plt.close()
    
    # Create individual plots if requested
    if create_individual_plots:
        print(f"Creating individual ridge plots for each controller...")
        
        # Create individual plots for each controller
        for method in filtered_noise_data.keys():
            controller_data = df_ridge[df_ridge['Controller'] == method]
            
            if len(controller_data) == 0:
                print(f"No data for controller {method}, skipping...")
                continue
                
            # Calculate RMSE range for this controller
            controller_rmse = controller_data['RMSE'].values
            rmse_min = max(0, np.min(controller_rmse) * 0.95)
            rmse_max = np.max(controller_rmse) * 1.05
            
            print(f"Creating individual plot for {method}, RMSE range: [{rmse_min:.4f}, {rmse_max:.4f}]")
            
            # Create figure for this controller - single subplot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Get base color for this controller from plot_colors
            base_color = plot_colors.get(method, 'gray')
            
            # Create color gradient for noise levels (darker = higher noise)
            noise_colors = create_color_gradient(base_color, len(selected_noise_levels), reverse=False)
            
            # Plot ridge for each noise level (reverse order so highest noise is at top)
            y_offset = 0
            ridge_height = 0.8
            
            for j, (noise_idx, noise_level) in enumerate(zip(reversed(selected_noise_indices), reversed(selected_noise_levels))):
                noise_data_subset = controller_data[controller_data['NoiseIndex'] == noise_idx]
                
                if len(noise_data_subset) > 0:
                    rmse_values = noise_data_subset['RMSE'].values
                    
                    # Color assignment: higher noise levels get darker colors
                    color_idx = len(selected_noise_levels) - 1 - j
                    current_color = noise_colors[color_idx]
                    
                    if len(rmse_values) > 1:
                        # Create KDE for this noise level
                        kde = gaussian_kde(rmse_values)
                        x_range = np.linspace(max(rmse_min, np.min(rmse_values) * 0.99), 
                                            min(rmse_max, np.max(rmse_values) * 1.01), 200)
                        density = kde(x_range)
                        
                        # Normalize density for consistent ridge height
                        density = density / np.max(density) * ridge_height
                        
                        # Plot the ridge
                        ax.fill_between(x_range, y_offset, y_offset + density, 
                                       color=current_color, alpha=0.7, zorder=2)
                        ax.plot(x_range, y_offset + density, 
                               color=current_color, linewidth=2, zorder=3)
                        
                        # Add mean line
                        mean_rmse = np.mean(rmse_values)
                        ax.axvline(x=mean_rmse, ymin=y_offset/(len(selected_noise_levels)), 
                                  ymax=(y_offset+ridge_height)/(len(selected_noise_levels)),
                                  color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=4)
                        
                        # Add noise level label (positioned on the left)
                        if noise_option == 'param':
                            label_text = f'Param {noise_level:.2f}'
                        else:
                            label_text = f'Noise {noise_level}'
                        
                        ax.text(rmse_min + (rmse_max - rmse_min) * 0.02, y_offset + ridge_height/2, label_text,
                               fontsize=11, va='center', ha='left', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=current_color, alpha=0.9))
                        
                        # Add mean value annotation (positioned on the right)
                        ax.text(rmse_min + (rmse_max - rmse_min) * 0.98, y_offset + ridge_height/2, 
                               f'μ={mean_rmse:.3f}',
                               fontsize=10, va='center', ha='right',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
                    
                    else:
                        # Single point - show as vertical line
                        ax.axvline(x=rmse_values[0], ymin=y_offset/(len(selected_noise_levels)), 
                                  ymax=(y_offset+ridge_height)/(len(selected_noise_levels)),
                                  color=current_color, linewidth=4, alpha=0.8, zorder=2)
                        
                        # Add label for single point
                        if noise_option == 'param':
                            label_text = f'Param {noise_level:.2f}'
                        else:
                            label_text = f'Noise {noise_level}'
                        
                        ax.text(rmse_min + (rmse_max - rmse_min) * 0.02, y_offset + ridge_height/2, label_text,
                               fontsize=11, va='center', ha='left', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=current_color, alpha=0.9))
                    
                    y_offset += 1
            
            # Customize plot
            ax.set_xlim(rmse_min, rmse_max)
            ax.set_ylim(-0.2, len(selected_noise_levels) + 0.2)
            ax.set_xlabel('RMSE [m]', fontsize=14, weight='bold')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(2)
            
            # Set title
            noise_type_titles = {
                'obs_noise': 'Observation Noise',
                'proc_noise': 'Process Noise', 
                'param': 'Parameter Uncertainty'
            }
            noise_type = noise_type_titles.get(noise_option, noise_option)
            ax.set_title(f'{method} - {noise_type} Robustness', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Add a subtitle with noise levels
            if noise_option == 'param':
                level_str = ', '.join([f'{x:.2f}' for x in selected_noise_levels])
                subtitle = f'Parameter scales: {level_str}'
            else:
                level_str = ', '.join([str(x) for x in selected_noise_levels])
                subtitle = f'Noise levels: {level_str}'
            
            ax.text(0.5, -0.15, subtitle, 
                   transform=ax.transAxes, ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            
            # Save individual plot for this controller
            safe_method_name = method.replace(' ', '_').replace('-', '_')
            individual_save_name = f"ridge_plot_{safe_method_name}_{noise_option}_individual"
            individual_save_path = script_dir / 'noise' / f'{individual_save_name}.png'
            individual_save_path.parent.mkdir(exist_ok=True)
            plt.savefig(individual_save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
            plt.savefig(individual_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
            print(f"Individual ridge plot for {method} saved as {individual_save_path}")
            plt.close()
    
    return df_ridge

def create_ridge_plot_custom_levels(noise_levels_to_plot, controllers_to_plot=None):
    """
    Create ridge plots for specific noise levels.
    
    Args:
        noise_levels_to_plot: List of noise levels to include in the plot
        controllers_to_plot: List of controller names to plot. If None, plot all controllers.
    """
    # Find indices for the requested noise levels
    selected_indices = []
    for level in noise_levels_to_plot:
        try:
            idx = noise_scale.index(level)
            selected_indices.append(idx)
        except ValueError:
            print(f"Warning: Noise level {level} not found in noise_scale")
    
    if not selected_indices:
        print("No valid noise levels found")
        return None
    
    return create_ridge_plot_at_noise_levels(selected_indices, controllers_to_plot)

# Main execution
if __name__ == "__main__":
    print("Creating ridge plots for different noise levels...")
    
    # Example controller selections (uncomment to use specific controllers)
    # Note: Available controllers depend on method_type:
    # Control methods: 'Linear MPC', 'Nonlinear MPC', 'GP-MPC', 'iLQR', 'LQR', 'Geometric Control', 'F-MPC'
    # RL methods: 'PPO', 'SAC', 'DPPO', 'PPO-MPC'
    
    if method_type == 'control':
        controllers_to_plot = None  # Plot all control methods  
        # controllers_to_plot = ['LQR', 'F-MPC']  # Only specific control methods
        # controllers_to_plot = ['iLQR', 'LQR']  # Only optimal control methods
        # controllers_to_plot = ['GP-MPC']  # Single controller
    elif method_type == 'rl':
        controllers_to_plot = None  # Plot all RL methods
        # controllers_to_plot = ['PPO', 'SAC']  # Only specific RL methods
        # controllers_to_plot = ['PPO-MPC']  # Single RL controller
    else:  # method_type == 'all'
        controllers_to_plot = None  # Plot all methods
        # controllers_to_plot = ['LQR', 'F-MPC', 'PPO', 'SAC']  # Mixed selection
    
    # Create ridge plot with auto-selected representative levels
    if noise_option == 'param':
        # For parametric uncertainty, create individual plots for each algorithm with specific levels
        print("\nCreating individual ridge plots for each algorithm with parametric uncertainty...")
        # Use specific noise levels for parametric uncertainty individual plots
        custom_param_levels = [0, 1.0, 2.0, 3.0, 4.0]
        param_indices = []
        for level in custom_param_levels:
            try:
                idx = noise_scale.index(level)
                param_indices.append(idx)
            except ValueError:
                print(f"Warning: Noise level {level} not found in noise_scale")
        
        if param_indices:
            df_ridge = create_ridge_plot_at_noise_levels(selected_noise_indices=param_indices, 
                                                       controllers_to_plot=controllers_to_plot, 
                                                       create_individual_plots=True)
        else:
            print("No valid parameter levels found, using auto-selected levels")
            df_ridge = create_ridge_plot_at_noise_levels(controllers_to_plot=controllers_to_plot, create_individual_plots=True)
    else:
        # For other noise types, create standard combined ridge plot
        df_ridge = create_ridge_plot_at_noise_levels(controllers_to_plot=controllers_to_plot)
    
    # Example: Create a custom plot for specific noise levels and controllers
    if noise_option == 'obs_noise':
        print("\nCreating custom ridge plot for specific observation noise levels...")
        custom_levels = [1, 5, 20, 50, 100]
        # Example: Plot specific controllers for these noise levels based on method type
        if method_type == 'control':
            custom_controllers = ['GP-MPC', 'Linear MPC']
        elif method_type == 'rl':
            custom_controllers = ['PPO', 'SAC']
        else:  # method_type == 'all'
            custom_controllers = ['GP-MPC', 'PPO']  # Mix of control and RL
        create_ridge_plot_custom_levels(custom_levels, controllers_to_plot=custom_controllers)
    elif noise_option == 'proc_noise':
        print("\nCreating custom ridge plot for specific process noise levels...")
        custom_levels = [1, 10, 30, 60, 100]
        if method_type == 'control':
            custom_controllers = ['LQR', 'F-MPC']
        elif method_type == 'rl':
            custom_controllers = ['DPPO', 'PPO-MPC']
        else:  # method_type == 'all'
            custom_controllers = ['LQR', 'PPO']  # Mix of control and RL
        create_ridge_plot_custom_levels(custom_levels, controllers_to_plot=custom_controllers)
    elif noise_option == 'param':
        print("\nCreating custom ridge plot for specific parameter uncertainty levels...")
        custom_levels = [0, 1.0, 2.0, 3.0, 4.0]
        if method_type == 'control':
            custom_controllers = ['GP-MPC', 'iLQR']
        elif method_type == 'rl':
            custom_controllers = ['SAC', 'PPO-MPC']
        else:  # method_type == 'all'
            custom_controllers = ['GP-MPC', 'SAC']  # Mix of control and RL
        create_ridge_plot_custom_levels(custom_levels, controllers_to_plot=custom_controllers)
