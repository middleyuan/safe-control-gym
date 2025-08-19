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
    import plot_colors, tag_ctrl_list

# script dir
script_dir = Path(__file__).parent.resolve()

max_seed = 10
metric_name = 'metrics.txt'

if len(sys.argv) > 1:
    noise_option = sys.argv[1]
else:
    noise_option = 'obs_noise'
    # noise_option = 'proc_noise'
    # noise_option = 'param'
print('noise_option', noise_option)

# Load data
linear_mpc_data = np.load(script_dir / f'../data/linear_mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
mpc_data = np.load(script_dir / f'../data/mpc_acados_{noise_option}_results.npy', allow_pickle=True).item()
gpmpc_data = np.load(script_dir / f'../data/gpmpc_acados_TP_{noise_option}_results.npy', allow_pickle=True).item()
ilqr_data = np.load(script_dir / f'../data/ilqr_{noise_option}_results.npy', allow_pickle=True).item()
lqr_data = np.load(script_dir / f'../data/lqr_{noise_option}_results.npy', allow_pickle=True).item()
pid_data = np.load(script_dir / f'../data/pid_{noise_option}_results.npy', allow_pickle=True).item()
fmpc_data = np.load(script_dir / f'../data/fmpc_{noise_option}_results.npy', allow_pickle=True).item()


noise_data = {
    'Linear MPC': linear_mpc_data, 
    'Nonlinear MPC': mpc_data, 
    'GP-MPC': gpmpc_data,
    'iLQR': ilqr_data,
    'LQR': lqr_data,
    'Geometric Control': pid_data,
    'F-MPC': fmpc_data,
} 

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

def create_ridge_plot_at_noise_levels(selected_noise_indices=None, controllers_to_plot=None):
    """
    Create ridge plots showing RMSE spread at specific noise levels.
    Each subplot shows one controller's RMSE distribution at the selected noise levels.
    
    Args:
        selected_noise_indices: List of indices into noise_scale to plot. 
                               If None, will automatically select representative levels.
        controllers_to_plot: List of controller names to plot. If None, plot all controllers.
                            Available controllers: 'Linear MPC', 'Nonlinear MPC', 'GP-MPC', 
                            'iLQR', 'LQR', 'Geometric Control', 'F-MPC'
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
    ridge_save_name = f"ridge_plot_noise_levels_{noise_option}"
    ridge_save_path = script_dir / 'noise' / f'{ridge_save_name}.png'
    ridge_save_path.parent.mkdir(exist_ok=True)
    plt.savefig(ridge_save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.savefig(ridge_save_path.with_suffix('.pdf'), bbox_inches="tight", pad_inches=0.1)
    print(f"Ridge plot saved as {ridge_save_path}")
    plt.show()
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
    controllers_to_plot = ['LQR', 'F-MPC']  # Only optimal control methods
    # controllers_to_plot = ['iLQR', 'LQR']  # Only optimal control methods
    # controllers_to_plot = ['GP-MPC']  # Single controller
    # controllers_to_plot = None  # Plot all controllers (default)
    
    # Create ridge plot with auto-selected representative levels
    df_ridge = create_ridge_plot_at_noise_levels(controllers_to_plot=controllers_to_plot)
    
    # Example: Create a custom plot for specific noise levels and controllers
    if noise_option == 'obs_noise':
        print("\nCreating custom ridge plot for specific observation noise levels...")
        custom_levels = [0, 5, 20, 50, 100]
        # Example: Plot only GP-MPC and Linear MPC for these noise levels
        # create_ridge_plot_custom_levels(custom_levels, controllers_to_plot=['GP-MPC', 'Linear MPC'])
        create_ridge_plot_custom_levels(custom_levels, controllers_to_plot=controllers_to_plot)
    elif noise_option == 'proc_noise':
        print("\nCreating custom ridge plot for specific process noise levels...")
        custom_levels = [0, 10, 30, 60, 100]
        create_ridge_plot_custom_levels(custom_levels, controllers_to_plot=controllers_to_plot)
    elif noise_option == 'param':
        print("\nCreating custom ridge plot for specific parameter uncertainty levels...")
        custom_levels = [0, 0.5, 1.0, 2.0, 4.0]
        create_ridge_plot_custom_levels(custom_levels, controllers_to_plot=controllers_to_plot)
