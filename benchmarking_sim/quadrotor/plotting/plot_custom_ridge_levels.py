"""
Custom ridge plot for selected controllers at different noise levels.

This script creates ridge plots showing RMSE distributions for specific controllers
at different noise levels, allowing comparison of robustness across selected methods.

Usage:
    python plot_custom_ridge_levels.py <noise_type> <controller1> <controller2> ... [--include-dr controller_dr1 ...]
    
Arguments:
    noise_type: 'obs_noise', 'proc_noise', or 'param'
    controllers: Space-separated list of controller names
    --include-dr: Optional flag followed by space-separated list of controllers to include DR variants
    
Available Controllers:
    Model-based: linear_mpc_acados, mpc_acados, gpmpc_acados_TP, ilqr, lqr, pid, fmpc
    RL: ppo, sac, dppo, ppo_mpc
    
Examples:
    python plot_custom_ridge_levels.py obs_noise lqr fmpc mpc_acados sac --include-dr sac
    python plot_custom_ridge_levels.py param ppo sac dppo --include-dr ppo sac dppo
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

from benchmarking_sim.quadrotor.benchmark_util.utils import plot_colors

# script dir
script_dir = Path(__file__).parent.resolve()

def create_color_gradient(base_color, n_levels, reverse=False):
    """Create a color gradient from original color to darker based on the base color."""
    
    # Convert base color to RGB if it's a named color
    if isinstance(base_color, str):
        if base_color.startswith('#'):
            base_rgb = mcolors.hex2color(base_color)
        else:
            base_rgb = mcolors.to_rgb(base_color)
    else:
        base_rgb = base_color
    
    # Create gradient from original color (1.0 intensity) to darker (0.5 intensity)
    # Changed from 0.3 to 0.5 to make darkest colors lighter
    alphas = np.linspace(1.0, 0.5, n_levels)
    if reverse:
        alphas = alphas[::-1]
    
    colors = []
    for alpha in alphas:
        # Make colors darker by reducing intensity
        dark_color = tuple(base_rgb[i] * alpha for i in range(3))
        colors.append(dark_color)
    
    return colors

def load_controller_data(controller_name, noise_type, include_dr=False):
    """Load data for a specific controller."""
    
    # Controller name mappings for display
    display_name_map = {
        'linear_mpc_acados': 'Linear MPC',
        'mpc_acados': 'Nonlinear MPC', 
        'gpmpc_acados_TP': 'GP-MPC',
        'ilqr': 'iLQR',
        'lqr': 'LQR',
        'pid': 'Geometric Control',
        'fmpc': 'F-MPC',
        'ppo': 'PPO',
        'sac': 'SAC',
        'dppo': 'DPPO',
        'ppo_mpc': 'PPO-MPC',
    }
    
    controller_data = {}
    
    # Load regular controller data
    file_path = script_dir / f'../data/{controller_name}_{noise_type}_results.npy'
    if file_path.exists():
        try:
            data = np.load(file_path, allow_pickle=True).item()
            display_name = display_name_map.get(controller_name, controller_name.upper())
            controller_data[display_name] = data
            print(f"  Loaded: {display_name}")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    else:
        print(f"  Warning: {file_path} not found")
    
    # Load domain randomization data if requested and available
    if include_dr and controller_name in ['ppo', 'sac', 'dppo']:
        dr_file = script_dir / f'../data/{controller_name}_domain_rand_robustness.npy'
        if dr_file.exists():
            try:
                dr_data = np.load(dr_file, allow_pickle=True).item()
                
                # Map noise type names
                noise_type_map = {
                    'obs_noise': 'obs_noise',
                    'proc_noise': 'proc_noise', 
                    'param': 'param'
                }
                
                if noise_type_map[noise_type] in dr_data['robustness']:
                    robustness_data = dr_data['robustness'][noise_type_map[noise_type]]
                    
                    # Convert to the same format as normal robustness data
                    noise_scales = sorted(robustness_data.keys())
                    rmse_means = []
                    rmse_stds = []
                    
                    for scale in noise_scales:
                        scale_data = robustness_data[scale]
                        rmse_means.append(scale_data['rmse']['mean'])
                        rmse_stds.append(scale_data['rmse']['std'])
                    
                    # Calculate degradation relative to baseline
                    if noise_scales and noise_scales[0] == 0.0:
                        baseline_rmse = rmse_means[0]
                    else:
                        baseline_rmse = min(rmse_means) if rmse_means else 1.0
                    
                    rmse_degradation_mean = [(rmse / baseline_rmse) * 100 for rmse in rmse_means]
                    rmse_degradation_std = [(std / baseline_rmse) * 100 for std in rmse_stds]
                    
                    display_name = display_name_map.get(controller_name, controller_name.upper()) + " (DR)"
                    
                    controller_data[display_name] = {
                        'noise_factor': np.array(noise_scales),
                        'rmse_mean': np.array(rmse_means),
                        'rmse_std': np.array(rmse_stds),
                        'rmse_degradation_mean': np.array(rmse_degradation_mean),
                        'rmse_degradation_std': np.array(rmse_degradation_std),
                        'baseline_rmse': baseline_rmse,
                        'controller': f"{controller_name}_domain_rand",
                        'noise_type': noise_type
                    }
                    
                    print(f"  Loaded: {display_name}")
                else:
                    print(f"  No {noise_type} data found in DR file for {controller_name}")
            except Exception as e:
                print(f"  Error loading DR data for {controller_name}: {e}")
        else:
            print(f"  Warning: DR file not found for {controller_name}")
    
    return controller_data

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

def create_custom_ridge_plot(noise_data, noise_type, noise_scale, selected_noise_indices=None):
    """Create ridge plot for selected controllers at different noise levels."""
    
    # Auto-select representative noise levels if not specified
    if selected_noise_indices is None:
        if len(noise_scale) >= 8:
            # Select specific indices to include noise level 1 and show progression
            # For obs_noise/proc_noise: [0, 1, 6, 16, 45, 100] -> indices [0, 1, 6, 13, 20, 26]
            if noise_type in ['obs_noise', 'proc_noise']:
                # Find indices for specific noise levels we want to show
                target_levels = [1, 5, 20, 50, 100]
                selected_noise_indices = []
                for target in target_levels:
                    if target in noise_scale:
                        selected_noise_indices.append(noise_scale.index(target))
            else:
                # For other noise types, use evenly spaced indices
                # selected_noise_indices = [0, len(noise_scale)//4, len(noise_scale)//2, 
                #                         3*len(noise_scale)//4, len(noise_scale)-1]
                target_levels = [0, 1, 2, 3, 4]
                selected_noise_indices = []
                for target in target_levels:
                    if target in noise_scale:
                        selected_noise_indices.append(noise_scale.index(target))
        else:
            selected_noise_indices = list(range(len(noise_scale)))
    
    selected_noise_levels = [noise_scale[i] for i in selected_noise_indices]
    print(f"Selected noise levels: {selected_noise_levels}")
    
    # Generate synthetic RMSE samples for each controller at each noise level
    ridge_data = []
    
    for method in noise_data.keys():
        if 'rmse_mean' in noise_data[method] and 'rmse_std' in noise_data[method]:
            rmse_means = noise_data[method]['rmse_mean']
            rmse_stds = noise_data[method]['rmse_std']
            
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
    n_controllers = len(noise_data)
    fig, axes = plt.subplots(n_controllers, 1, figsize=(6, n_controllers * 2), 
                            sharex=True, gridspec_kw={'hspace': 0.45}, dpi=1200)
    
    if n_controllers == 1:
        axes = [axes]
    
    # Extended plot colors for DR variants
    extended_colors = plot_colors.copy()
    for method in noise_data.keys():
        if '(DR)' in method:
            base_method = method.replace(' (DR)', '')
            if base_method in plot_colors:
                extended_colors[method] = plot_colors[base_method]
    
    for i, (method, ax) in enumerate(zip(noise_data.keys(), axes)):
        controller_data = df_ridge[df_ridge['Controller'] == method]
        
        # Get base color for this controller
        base_color = extended_colors.get(method, 'gray')
        
        # Create color gradient for noise levels (original color to darker)
        noise_colors = create_color_gradient(base_color, len(selected_noise_levels), reverse=False)
        
        if len(controller_data) > 0:
            # Track y-position for ridge effect - start from top (lowest noise first)
            y_spacing = 1.0
            max_y = len(selected_noise_levels) * y_spacing
            
            # Reverse the order: lowest noise at top, highest at bottom
            for j, (noise_idx, noise_level) in enumerate(zip(selected_noise_indices, selected_noise_levels)):
                subset = controller_data[controller_data['NoiseLevel'] == noise_level]
                
                if len(subset) > 10:  # Need sufficient data for KDE
                    try:
                        rmse_values = subset['RMSE'].values
                        mean_rmse = np.mean(rmse_values)
                        
                        # Create KDE
                        kde = gaussian_kde(rmse_values)
                        x_range = np.linspace(global_rmse_min, global_rmse_max, 200)
                        density = kde(x_range)
                        
                        # Normalize density for plotting
                        density = density / np.max(density) * 0.8  # Scale to fit subplot
                        
                        # Calculate y_offset from top (reverse order)
                        y_offset = max_y - (j + 1) * y_spacing
                        
                        # Plot filled curve
                        color = noise_colors[j]  # j=0 is lowest noise (original color)
                        linestyle = '--' if '(DR)' in method else '-'
                        
                        ax.fill_between(x_range, y_offset, y_offset + density, 
                                      alpha=0.6, color=color, label=f'Noise={noise_level}')
                        ax.plot(x_range, y_offset + density, color=color, 
                               linewidth=1.5, linestyle=linestyle)
                        
                        # Add grey dashed line at mean
                        ridge_peak_y = y_offset + np.max(density)
                        ax.plot([mean_rmse, mean_rmse], [y_offset, ridge_peak_y], 
                               color='dimgray', linestyle='--', linewidth=1.5, alpha=0.8)
                        
                        # Add mean value annotation in a small box
                        ax.text(global_rmse_max * 0.95, y_offset + np.max(density) * 0.5, 
                               f'$\mu$={mean_rmse:.3f}', 
                               fontsize=8, ha='right', va='center',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", 
                                        edgecolor="grey", alpha=0.8))
                        
                    except Exception as e:
                        print(f"Warning: Could not create KDE for {method} at noise level {noise_level}: {e}")
        
        # Customize subplot
        ax.set_ylabel('')  # Remove the y-label
        ax.set_ylim(0, max_y)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        
        # Add controller name at the top center with controller's color background
        ax.text(0.5, 1.15, method, transform=ax.transAxes, 
               fontsize=11, ha='center', va='center', weight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor=base_color, alpha=0.8, edgecolor='black'))
        
        # Add legend for this controller outside the plot area (to the right)
        handles, labels = ax.get_legend_handles_labels()
        if handles:  # Only create legend if there are handles
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 0.94), loc='upper left', fontsize=8)
    
    # Set common x-label and limits
    axes[-1].set_xlabel('RMSE (m)', fontsize=12)
    for ax in axes:
        ax.set_xlim(global_rmse_min, global_rmse_max)
    
    # Create title
    noise_title_map = {
        'obs_noise': 'Observation Noise',
        'proc_noise': 'Process Noise',
        'param': 'Parametric Uncertainty'
    }
    
    title = f'RMSE Distributions at Different Levels of {noise_title_map.get(noise_type, noise_type)}'
    fig.suptitle(title, fontsize=14, y=0.95, x=0.65, ha='center', va='top') # centered horizontally at the top
    
    # Save plot
    controller_names_str = "_".join([name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") for name in noise_data.keys()])
    save_name = f"custom_ridge_levels_{noise_type}_{controller_names_str}"
    save_path = script_dir / 'custom_plots' / f'{save_name}.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
    print(f"Custom ridge levels plot saved as {save_path}")
    plt.show()
    
    return df_ridge

def main():
    """Main function to parse arguments and generate custom ridge plot."""
    
    if len(sys.argv) < 3:
        print("Usage: python plot_custom_ridge_levels.py <noise_type> <controller1> <controller2> ... [--include-dr controller_dr1 ...]")
        print("Example: python plot_custom_ridge_levels.py obs_noise lqr fmpc mpc_acados sac --include-dr sac")
        return
    
    # Parse arguments
    args = sys.argv[1:]
    noise_type = args[0]
    
    if noise_type not in ['obs_noise', 'proc_noise', 'param']:
        print(f"Error: Invalid noise type '{noise_type}'. Must be one of: obs_noise, proc_noise, param")
        return
    
    # Find --include-dr flag
    dr_flag_idx = None
    if '--include-dr' in args:
        dr_flag_idx = args.index('--include-dr')
    
    # Extract controller lists
    if dr_flag_idx is not None:
        controllers = args[1:dr_flag_idx]
        dr_controllers = args[dr_flag_idx+1:]
    else:
        controllers = args[1:]
        dr_controllers = []
    
    if not controllers:
        print("Error: No controllers specified")
        return
    
    print(f"Noise type: {noise_type}")
    print(f"Controllers: {controllers}")
    print(f"DR Controllers: {dr_controllers}")
    
    # Define noise scales
    if noise_type in ['obs_noise']:
        noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       12, 14, 16, 18, 20, 25, 30, 35, 40, 
                       45, 50, 60, 70, 80, 90, 100]
    elif noise_type == 'proc_noise':
        noise_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       12, 14, 16, 18, 20, 25, 30, 35, 40, 
                       45, 50, 60, 70, 80, 90, 100]
    elif noise_type == 'param':
        noise_scale = [0, 0.01, 0.02, 0.05, 0.1, 
                       0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 
                       1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 
                       2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0]
        noise_scale.sort()
    
    # Load data for all requested controllers
    all_data = {}
    
    print("\nLoading controller data...")
    for controller in controllers:
        include_dr = controller in dr_controllers
        controller_data = load_controller_data(controller, noise_type, include_dr)
        all_data.update(controller_data)
    
    if not all_data:
        print("No data loaded. Please check controller names and ensure data files exist.")
        print("Run process_experiment_data.py first to generate the required data files.")
        return
    
    # Align all data to common noise scale
    for method in all_data.keys():
        all_data[method] = align_data_to_scale(all_data[method], noise_scale)
    
    print(f"\nLoaded {len(all_data)} controller variants")
    
    # Generate custom ridge plot
    create_custom_ridge_plot(all_data, noise_type, noise_scale)

if __name__ == '__main__':
    main()
