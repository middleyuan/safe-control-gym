import os
import yaml
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import ast

from safe_control_gym.hyperparameters.hpo_utils import get_smallest_seed_folder, load_trials_data

# Define the base directory
base_dir = 'examples/hpo/hpo'  # Change this if needed
algorithms = ['pid', 'lqr', 'ilqr']
trials = 50  # Number of trials for HPO
# scenarios = ['basic', 'ob_ns=5', 'proc_ns=3', 'ob_ns=5_proc_ns=3', 'dr']  # List your scenarios here
scenarios = ['basic']
metric_weights = [0.6, 0.4]  # Weights for combining RMSE and RMS Action Change

# Function to load hand-tuned performance data
def load_handtune_data(algorithm, folder='vizier'):
    try:
        seed_folder = get_smallest_seed_folder(algorithm, folder, base_dir)
        file_path = os.path.join(base_dir, algorithm, folder, seed_folder, 'hpo', 'warmstart_trial_value.txt')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                metric_dict = ast.literal_eval(content)
                return np.mean(metric_dict['exponentiated_rmse']), np.mean(metric_dict['exponentiated_rms_action_change'])
    except:
        pass
    return None, None

# Plot Hyperparameter Evaluation Over Trials
def plot_hpo_evaluation(trials, algorithms, packages=['optuna', 'vizier'], target_name='Normalized Reward'):
    plt.style.use('ggplot')  # Use ggplot style for consistent appearance
    num_algorithms = len(algorithms)
    num_packages = len(packages)
    fig, axes = plt.subplots(num_packages, num_algorithms, figsize=(40, 10), sharex=True, sharey=True)
    fig.suptitle('Optimization History Plot for Hyperparameter Tuning', fontsize=16)
    cmap = plt.get_cmap('tab10')  # Colormap for distinguishing algorithms

    for row, package in enumerate(packages):
        for col, algorithm in enumerate(algorithms):
            ax = axes[row, col]
            ax.set_title(f'{package} - {algorithm}')
            ax.set_xlabel('Trial Number')
            ax.set_ylabel(target_name)

            # Load trial data
            trials_data = load_trials_data(algorithm, package, base_dir)
            if trials_data is not None:
                trial_numbers = trials_data['number'][:trials]
                if 'values_0' in trials_data.keys():
                    values = trials_data['values_0'][:trials]
                elif 'exponentiated_rmse' in trials_data.keys():
                    values = trials_data['exponentiated_rmse'][:trials]

                # Scatter plot for individual trials
                ax.scatter(trial_numbers, values, color=cmap(col), alpha=0.8, label=f'{package} - {algorithm}')

                # Optionally, plot best values as cumulative maximum
                best_values = np.maximum.accumulate(values)
                ax.plot(trial_numbers, best_values, color=cmap(col), alpha=0.6, linestyle='--', label='Best Values')

            ax.legend()

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('hpo_optimization.png')

# Box Plot Comparison of Performance
def plot_performance_comparison(algorithms, packages=['optuna', 'vizier']):
    data = []
    data_rms = []

    for algorithm in algorithms:
        # Hand-tuned data
        handtune_ob1, handtune_ob2 = load_handtune_data(algorithm)
        if handtune_ob1 is not None:
            data.append([algorithm, 'Hand Tuned', handtune_ob1, handtune_ob2])
            data_rms.append([algorithm, 'Hand Tuned', -np.log(handtune_ob1), -np.log(handtune_ob2)])

        # HPO data for each package
        for package in packages:
            trials_data = load_trials_data(algorithm, package, base_dir)
            if trials_data is not None:
                if 'values_0' in trials_data.keys():
                    norm_rmse = trials_data['values_0']
                elif 'exponentiated_rmse' in trials_data.keys():
                    norm_rmse = trials_data['exponentiated_rmse']
                if 'values_1' in trials_data.keys():
                    norm_rms = trials_data['values_1']
                elif 'exponentiated_rms_action_change' in trials_data.keys():
                    norm_rms = trials_data['exponentiated_rms_action_change']
                combined_norm_metric = metric_weights[0] * norm_rmse + metric_weights[1] * norm_rms
                index = combined_norm_metric.idxmax()
                data.append([algorithm, package, norm_rmse[index], norm_rms[index]])
                if 'values_0' in trials_data.keys():
                    rmse = -np.log(trials_data['values_0'])
                elif 'exponentiated_rmse' in trials_data.keys():
                    rmse = -np.log(trials_data['exponentiated_rmse'])
                if 'values_1' in trials_data.keys():
                    rms = -np.log(trials_data['values_1'])
                elif 'exponentiated_rms_action_change' in trials_data.keys():
                    rms = -np.log(trials_data['exponentiated_rms_action_change'])
                combined_metric = rmse + rms
                index = combined_metric.idxmin()
                data_rms.append([algorithm, package, rmse[index], rms[index]])
            else:
                raise ValueError(f'No trials data found for {algorithm} - {package}')

    # Convert to DataFrame for plotting
    df = pd.DataFrame(data, columns=['Algorithm', 'Method', 'Normalized Tracking Reward', 'Normalized Action Change Reward'])
    rmse_palette = ['#1f77b4', '#d62728', '#9467bd']
    rms_palette = ['#a1c9f4', '#ff9f9b', '#d0bbff']

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Algorithm', y='Normalized Tracking Reward', hue='Method', data=df[['Algorithm', 'Method', 'Normalized Tracking Reward']], palette=rmse_palette)
    sns.move_legend(plt.gca(), 'lower left')
    plt.title('Performance Comparison of Hand-Tuned, Optuna, and Vizier')
    plt.savefig('hpo_performance_comparison.png')

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Algorithm', y='Normalized Action Change Reward', hue='Method', data=df[['Algorithm', 'Method', 'Normalized Action Change Reward']])
    sns.move_legend(plt.gca(), 'lower left')
    plt.title('Action Change Comparison of Hand-Tuned, Optuna, and Vizier')
    plt.savefig('hpo_action_change_comparison.png')

    df_rms = pd.DataFrame(data_rms, columns=['Algorithm', 'Method', 'RMSE', 'RMS Action Change'])

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Algorithm', y='RMSE', hue='Method', data=df_rms[['Algorithm', 'Method', 'RMSE']])
    sns.move_legend(plt.gca(), 'lower left')
    plt.title('RMSE Comparison of Hand-Tuned, Optuna, and Vizier')
    plt.savefig('hpo_rmse_comparison.png')

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Algorithm', y='RMS Action Change', hue='Method', data=df_rms[['Algorithm', 'Method', 'RMS Action Change']])
    sns.move_legend(plt.gca(), 'lower left')
    plt.title('RMS Action Change Comparison of Hand-Tuned, Optuna, and Vizier')
    plt.savefig('hpo_rms_action_change_comparison.png')


    plt.figure(figsize=(14, 8))
    # Aggregate data for stacking
    tracking_data = df.groupby(["Algorithm", "Method"])["Normalized Tracking Reward"].sum().unstack()
    action_data = df.groupby(["Algorithm", "Method"])["Normalized Action Change Reward"].sum().unstack()
    # Bar positions
    algorithms = tracking_data.index
    x = np.arange(len(algorithms))  # Numerical positions for algorithms
    # make spacing larger
    x = x * 2
    bar_width = 0.5  # Width of each bar group

    # Plot hand-tuned data
    plt.bar(x - bar_width, tracking_data["Hand Tuned"], label="Tracking Reward (Hand Tuned)", color=rmse_palette[0], width=bar_width)
    plt.bar(x - bar_width, action_data["Hand Tuned"], bottom=tracking_data["Hand Tuned"], label="Action Reward (Hand Tuned)", color=rms_palette[0], width=bar_width)

    # Plot optuna (stacked)
    plt.bar(x, tracking_data["optuna"], label="Tracking Reward (Optuna)", color=rmse_palette[1], width=bar_width)
    plt.bar(x, action_data["optuna"], bottom=tracking_data["optuna"], label="Action Reward (Optuna)", color=rms_palette[1], width=bar_width)

    # Plot vizier (stacked)
    plt.bar(x + bar_width, tracking_data["vizier"], label="Tracking Reward (Vizier)", color=rmse_palette[2], width=bar_width)
    plt.bar(x + bar_width, action_data["vizier"], bottom=tracking_data["vizier"], label="Action Reward (Vizier)", color=rms_palette[2], width=bar_width)

    # Add labels, legend, and title
    plt.xticks(x, algorithms)  # Rotate x-axis labels for clarity
    plt.xlabel("Algorithm")
    plt.ylabel("Normalized Rewards")
    plt.title("Performance Comparison of Hand-Tuned, Optuna, and Vizier")
    plt.legend(loc='lower left')
    plt.tight_layout()

    # Save the plot
    plt.savefig("grouped_stacked_hpo_chart.png")

    plt.figure(figsize=(14, 8))

    plt.bar(x - bar_width, -np.log(tracking_data["Hand Tuned"]), label="RMSE (Hand Tuned)", color=rmse_palette[0], width=bar_width)
    plt.bar(x - bar_width, -np.log(action_data["Hand Tuned"]), bottom=-np.log(tracking_data["Hand Tuned"]), label="RMS Action Change (Hand Tuned)", color=rms_palette[0], width=bar_width)

    # Plot optuna (stacked)
    plt.bar(x, -np.log(tracking_data["optuna"]), label="RMSE (Optuna)", color=rmse_palette[1], width=bar_width) 
    plt.bar(x, -np.log(action_data["optuna"]), bottom=-np.log(tracking_data["optuna"]), label="RMS Action Change (Optuna)", color=rms_palette[1], width=bar_width)

    # Plot vizier (stacked)
    plt.bar(x + bar_width, -np.log(tracking_data["vizier"]), label="RMSE (Vizier)", color=rmse_palette[2], width=bar_width)
    plt.bar(x + bar_width, -np.log(action_data["vizier"]), bottom=-np.log(tracking_data["vizier"]), label="RMS Action Change (Vizier)", color=rms_palette[2], width=bar_width)

    # Add labels, legend, and title
    plt.xticks(x, algorithms)  # Rotate x-axis labels for clarity
    plt.xlabel("Algorithm")
    plt.ylabel("RMSE and RMS Action Change")
    plt.title("RMSE and RMS Action Change Comparison of Hand-Tuned, Optuna, and Vizier")
    plt.legend()

    # Save the plot
    plt.savefig("grouped_stacked_hpo_rms_chart.png")

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert scalars to Python types
    elif isinstance(obj, str):
        try:
            # Try converting string representation of a list into a real list
            parsed = eval(obj)  # Use `json.loads(obj)` if the format is strict JSON
            if isinstance(parsed, list):
                return parsed
        except:
            pass  # If conversion fails, keep it as a string
    return obj

def is_pareto_efficient(costs):
    """
    Find the Pareto efficient points (non-dominated solutions).
    
    Args:
        costs: An (n_points, n_costs) array where each row is a point and each column is a cost/objective.
               For minimization problems (lower is better).
    
    Returns:
        A boolean array indicating which points are Pareto efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Check if point c is dominated by any other point
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True  # Keep self
    return is_efficient

def plot_pareto_front(algorithms, packages=['optuna', 'vizier']):
    """
    Plot Pareto front for multi-objective optimization results.
    Shows the trade-off between RMSE and RMS Action Change objectives.
    """
    plt.figure(figsize=(14, 10))
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms) * len(packages)))
    color_idx = 0
    
    all_points = []  # Store all points for global Pareto front
    all_labels = []  # Store corresponding labels
    has_multi_objective_data = False
    all_pareto_points = []  # Store all Pareto points for axis limit calculation
    hpo_selected_points = []  # Store HPO-selected points for each algorithm/package
    
    for algorithm in algorithms:
        for package in packages:
            trials_data = load_trials_data(algorithm, package, base_dir)
            if trials_data is not None:
                # Extract objectives (convert to costs for minimization)
                rmse_vals = None
                rms_action_vals = None
                
                if 'values_0' in trials_data.keys():
                    rmse_vals = -np.log(trials_data['values_0'])  # Convert to RMSE (lower is better)
                elif 'exponentiated_rmse' in trials_data.keys():
                    rmse_vals = -np.log(trials_data['exponentiated_rmse'])
                
                if 'values_1' in trials_data.keys():
                    rms_action_vals = -np.log(trials_data['values_1'])  # Convert to RMS Action Change (lower is better)
                elif 'exponentiated_rms_action_change' in trials_data.keys():
                    rms_action_vals = -np.log(trials_data['exponentiated_rms_action_change'])
                
                # Check if we have both objectives for multi-objective analysis
                if rmse_vals is not None and rms_action_vals is not None:
                    has_multi_objective_data = True
                    
                    # Create cost matrix for Pareto analysis
                    costs = np.column_stack([rmse_vals, rms_action_vals])
                    
                    # Find Pareto efficient points
                    pareto_mask = is_pareto_efficient(costs)
                    pareto_points = costs[pareto_mask]
                    
                    # Store points for global analysis
                    all_points.extend(costs.tolist())
                    all_labels.extend([f'{algorithm}_{package}'] * len(costs))
                    
                    # Store Pareto points for axis limit calculation
                    if len(pareto_points) > 0:
                        all_pareto_points.extend(pareto_points.tolist())
                    
                    # Find HPO-selected point based on weighted combination
                    # Convert back to normalized rewards for HPO criterion
                    norm_rmse = np.exp(-rmse_vals)  # Convert back from log
                    norm_rms_action = np.exp(-rms_action_vals)  # Convert back from log
                    combined_norm_metric = metric_weights[0] * norm_rmse + metric_weights[1] * norm_rms_action
                    hpo_selected_idx = combined_norm_metric.idxmax()
                    hpo_selected_point = [rmse_vals[hpo_selected_idx], rms_action_vals[hpo_selected_idx]]
                    hpo_score = combined_norm_metric[hpo_selected_idx]
                    hpo_selected_points.append((hpo_selected_point, f'{algorithm} ({package})', colors[color_idx], hpo_score))
                    
                    print(f"HPO selected point for {algorithm} ({package}): RMSE={hpo_selected_point[0]:.3f}, RMS_Action={hpo_selected_point[1]:.3f}, Score={hpo_score:.3f}")
                    
                    # Plot all points
                    plt.scatter(rmse_vals, rms_action_vals, 
                               color=colors[color_idx], alpha=0.3, s=30,
                               label=f'{algorithm} ({package}) - All trials')
                    
                    # Plot Pareto front points
                    if len(pareto_points) > 0:
                        plt.scatter(pareto_points[:, 0], pareto_points[:, 1], 
                                   color=colors[color_idx], s=100, marker='s', 
                                   edgecolors='black', linewidths=2,
                                   label=f'{algorithm} ({package}) - Pareto front')
                        
                        # Sort Pareto points for line connection
                        sorted_indices = np.argsort(pareto_points[:, 0])
                        sorted_pareto = pareto_points[sorted_indices]
                        plt.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
                                color=colors[color_idx], linestyle='--', alpha=0.7, linewidth=2)
                else:
                    print(f"Warning: {algorithm} ({package}) does not have multi-objective data for Pareto analysis")
                
                color_idx += 1
    
    if not has_multi_objective_data:
        print("No multi-objective data found. Pareto front analysis requires both RMSE and RMS Action Change objectives.")
        plt.close()
        return
    
    # Plot global Pareto front
    if all_points:
        all_costs = np.array(all_points)
        global_pareto_mask = is_pareto_efficient(all_costs)
        global_pareto_points = all_costs[global_pareto_mask]
        
        if len(global_pareto_points) > 0:
            # Sort for line connection
            sorted_indices = np.argsort(global_pareto_points[:, 0])
            sorted_global_pareto = global_pareto_points[sorted_indices]
            plt.plot(sorted_global_pareto[:, 0], sorted_global_pareto[:, 1], 
                    color='red', linewidth=3, marker='o', markersize=8,
                    label='Global Pareto Front', alpha=0.8)
    
    # Plot HPO-selected points
    for point, label, color, score in hpo_selected_points:
        plt.scatter(point[0], point[1], color=color, s=200, marker='*', 
                   edgecolors='black', linewidths=3, 
                   label=f'{label} - HPO Selected (Score: {score:.3f})', zorder=10)
    
    # Add text box explaining HPO selection
    textstr = f'HPO Selection Criterion:\n{metric_weights[0]:.2f} × Normalized_RMSE +\n{metric_weights[1]:.2f} × Normalized_RMS_Action'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Set axis limits based on Pareto front data with some padding
    if all_pareto_points:
        pareto_array = np.array(all_pareto_points)
        
        # Remove extreme outliers for better visualization (using IQR method)
        q1_x, q3_x = np.percentile(pareto_array[:, 0], [25, 75])
        q1_y, q3_y = np.percentile(pareto_array[:, 1], [25, 75])
        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y
        
        # Define outlier bounds (1.5 * IQR is standard, but we use 2.0 for more inclusive view)
        x_lower = q1_x - 2.0 * iqr_x
        x_upper = q3_x + 2.0 * iqr_x
        y_lower = q1_y - 2.0 * iqr_y
        y_upper = q3_y + 2.0 * iqr_y
        
        # Filter out extreme outliers
        mask = ((pareto_array[:, 0] >= x_lower) & (pareto_array[:, 0] <= x_upper) & 
                (pareto_array[:, 1] >= y_lower) & (pareto_array[:, 1] <= y_upper))
        
        if np.sum(mask) > 0:  # If we have points after filtering
            filtered_pareto = pareto_array[mask]
            x_min, x_max = filtered_pareto[:, 0].min(), filtered_pareto[:, 0].max()
            y_min, y_max = filtered_pareto[:, 1].min(), filtered_pareto[:, 1].max()
        else:  # Fallback to original data
            x_min, x_max = pareto_array[:, 0].min(), pareto_array[:, 0].max()
            y_min, y_max = pareto_array[:, 1].min(), pareto_array[:, 1].max()
        
        # Add padding (15% of the range for better visualization)
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = max(0.15 * x_range, 0.2)  # Minimum padding of 0.2
        y_padding = max(0.15 * y_range, 0.2)
        
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)
        
        print(f"Pareto front axis limits: x=[{x_min - x_padding:.2f}, {x_max + x_padding:.2f}], y=[{y_min - y_padding:.2f}, {y_max + y_padding:.2f}]")
    
    plt.xlabel('RMSE (log scale, lower is better)', fontsize=12)
    plt.ylabel('RMS Action Change (log scale, lower is better)', fontsize=12)
    plt.title(f'Pareto Front Analysis: RMSE vs RMS Action Change Trade-off\n(HPO Selection: {metric_weights[0]:.2f}×RMSE + {metric_weights[1]:.2f}×RMS_Action)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pareto_front_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved Pareto front analysis to pareto_front_analysis.png")

def plot_pareto_front_by_algorithm(algorithms, packages=['optuna', 'vizier']):
    """
    Plot separate Pareto front for each algorithm.
    """
    # First check if we have multi-objective data
    has_multi_objective_data = False
    valid_algorithms = []
    all_pareto_points_by_algo = []  # Store all Pareto points for consistent axis limits
    
    for algorithm in algorithms:
        for package in packages:
            trials_data = load_trials_data(algorithm, package, base_dir)
            if trials_data is not None:
                has_rmse = 'values_0' in trials_data.keys() or 'exponentiated_rmse' in trials_data.keys()
                has_rms_action = 'values_1' in trials_data.keys() or 'exponentiated_rms_action_change' in trials_data.keys()
                if has_rmse and has_rms_action:
                    has_multi_objective_data = True
                    if algorithm not in valid_algorithms:
                        valid_algorithms.append(algorithm)
                        
                    # Collect Pareto points for axis limit calculation
                    if 'values_0' in trials_data.keys():
                        rmse_vals = -np.log(trials_data['values_0'])
                    elif 'exponentiated_rmse' in trials_data.keys():
                        rmse_vals = -np.log(trials_data['exponentiated_rmse'])
                    
                    if 'values_1' in trials_data.keys():
                        rms_action_vals = -np.log(trials_data['values_1'])
                    elif 'exponentiated_rms_action_change' in trials_data.keys():
                        rms_action_vals = -np.log(trials_data['exponentiated_rms_action_change'])
                    
                    costs = np.column_stack([rmse_vals, rms_action_vals])
                    pareto_mask = is_pareto_efficient(costs)
                    pareto_points = costs[pareto_mask]
                    if len(pareto_points) > 0:
                        all_pareto_points_by_algo.extend(pareto_points.tolist())
    
    if not has_multi_objective_data:
        print("No multi-objective data found for algorithm-specific Pareto analysis.")
        return
    
    # Calculate consistent axis limits based on all Pareto points
    if all_pareto_points_by_algo:
        pareto_array = np.array(all_pareto_points_by_algo)
        
        # Remove extreme outliers for better visualization (using IQR method)
        q1_x, q3_x = np.percentile(pareto_array[:, 0], [25, 75])
        q1_y, q3_y = np.percentile(pareto_array[:, 1], [25, 75])
        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y
        
        # Define outlier bounds
        x_lower = q1_x - 2.0 * iqr_x
        x_upper = q3_x + 2.0 * iqr_x
        y_lower = q1_y - 2.0 * iqr_y
        y_upper = q3_y + 2.0 * iqr_y
        
        # Filter out extreme outliers
        mask = ((pareto_array[:, 0] >= x_lower) & (pareto_array[:, 0] <= x_upper) & 
                (pareto_array[:, 1] >= y_lower) & (pareto_array[:, 1] <= y_upper))
        
        if np.sum(mask) > 0:  # If we have points after filtering
            filtered_pareto = pareto_array[mask]
            x_min, x_max = filtered_pareto[:, 0].min(), filtered_pareto[:, 0].max()
            y_min, y_max = filtered_pareto[:, 1].min(), filtered_pareto[:, 1].max()
        else:  # Fallback to original data
            x_min, x_max = pareto_array[:, 0].min(), pareto_array[:, 0].max()
            y_min, y_max = pareto_array[:, 1].min(), pareto_array[:, 1].max()
        
        # Add padding (15% of the range for better visualization)
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = max(0.15 * x_range, 0.2)  # Minimum padding of 0.2
        y_padding = max(0.15 * y_range, 0.2)
        
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
        
        print(f"Algorithm-specific Pareto front axis limits: x=[{x_limits[0]:.2f}, {x_limits[1]:.2f}], y=[{y_limits[0]:.2f}, {y_limits[1]:.2f}]")
    else:
        x_limits = None
        y_limits = None
    
    fig, axes = plt.subplots(1, len(valid_algorithms), figsize=(6*len(valid_algorithms), 6), sharey=True)
    if len(valid_algorithms) == 1:
        axes = [axes]
    
    for idx, algorithm in enumerate(valid_algorithms):
        ax = axes[idx]
        colors = plt.cm.Set1(np.linspace(0, 1, len(packages)))
        hpo_selected_points_algo = []  # Store HPO-selected points for this algorithm
        
        for package_idx, package in enumerate(packages):
            trials_data = load_trials_data(algorithm, package, base_dir)
            if trials_data is not None:
                # Extract objectives
                rmse_vals = None
                rms_action_vals = None
                
                if 'values_0' in trials_data.keys():
                    rmse_vals = -np.log(trials_data['values_0'])
                elif 'exponentiated_rmse' in trials_data.keys():
                    rmse_vals = -np.log(trials_data['exponentiated_rmse'])
                
                if 'values_1' in trials_data.keys():
                    rms_action_vals = -np.log(trials_data['values_1'])
                elif 'exponentiated_rms_action_change' in trials_data.keys():
                    rms_action_vals = -np.log(trials_data['exponentiated_rms_action_change'])
                
                if rmse_vals is not None and rms_action_vals is not None:
                    # Create cost matrix
                    costs = np.column_stack([rmse_vals, rms_action_vals])
                    pareto_mask = is_pareto_efficient(costs)
                    pareto_points = costs[pareto_mask]
                    
                    # Find HPO-selected point
                    norm_rmse = np.exp(-rmse_vals)  # Convert back from log
                    norm_rms_action = np.exp(-rms_action_vals)  # Convert back from log
                    combined_norm_metric = metric_weights[0] * norm_rmse + metric_weights[1] * norm_rms_action
                    hpo_selected_idx = combined_norm_metric.idxmax()
                    hpo_selected_point = [rmse_vals[hpo_selected_idx], rms_action_vals[hpo_selected_idx]]
                    hpo_score = combined_norm_metric[hpo_selected_idx]
                    hpo_selected_points_algo.append((hpo_selected_point, package, colors[package_idx], hpo_score))
                    
                    # Plot all trials
                    ax.scatter(rmse_vals, rms_action_vals, 
                              color=colors[package_idx], alpha=0.5, s=30,
                              label=f'{package} - All trials')
                    
                    # Plot Pareto front
                    if len(pareto_points) > 0:
                        ax.scatter(pareto_points[:, 0], pareto_points[:, 1], 
                                  color=colors[package_idx], s=100, marker='s',
                                  edgecolors='black', linewidths=2,
                                  label=f'{package} - Pareto front')
                        
                        # Connect Pareto points
                        sorted_indices = np.argsort(pareto_points[:, 0])
                        sorted_pareto = pareto_points[sorted_indices]
                        ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
                               color=colors[package_idx], linestyle='--', alpha=0.7, linewidth=2)
        
        # Plot HPO-selected points for this algorithm
        for point, package, color, score in hpo_selected_points_algo:
            ax.scatter(point[0], point[1], color=color, s=200, marker='*', 
                      edgecolors='black', linewidths=3, 
                      label=f'{package} - HPO Selected ({score:.3f})', zorder=10)
        
        ax.set_xlabel('RMSE (log scale, lower is better)')
        if idx == 0:
            ax.set_ylabel('RMS Action Change (log scale, lower is better)')
        ax.set_title(f'{algorithm.upper()} Pareto Front\n(HPO: {metric_weights[0]:.2f}×RMSE + {metric_weights[1]:.2f}×RMS_Action)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set consistent axis limits for all subplots
        if x_limits and y_limits:
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
    
    plt.tight_layout()
    plt.savefig('pareto_front_by_algorithm.png', dpi=300, bbox_inches='tight')
    print("Saved algorithm-specific Pareto fronts to pareto_front_by_algorithm.png")

# Box plot comparison of performance in different HPO scenarios
def plot_performance_comparison_in_different_scenarios(algorithm, scenarios):
    data = []

    # HPO data for each scenario
    for scenario in scenarios:
        # Hand-tuned data
        handtune_ob1, handtune_ob2 = load_handtune_data(algorithm, scenario)
        if handtune_ob1 is not None:
            data.append([scenario, 'handtuned', handtune_ob1, handtune_ob2])
        trials_data = load_trials_data(algorithm, scenario, base_dir)
        if trials_data is not None:
            if 'values_0' in trials_data.keys():
                norm_rmse = trials_data['values_0']
            elif 'exponentiated_rmse' in trials_data.keys():
                norm_rmse = trials_data['exponentiated_rmse']
            if 'values_1' in trials_data.keys():
                norm_rms = trials_data['values_1']
            elif 'exponentiated_rms_action_change' in trials_data.keys():
                norm_rms = trials_data['exponentiated_rms_action_change']
            combined_norm_metric = metric_weights[0] * norm_rmse + metric_weights[1] * norm_rms
            index = combined_norm_metric.idxmax()
            data.append([scenario, 'optimized', norm_rmse[index], norm_rms[index]])

            # output the best hyperparameters in yaml format
            best_hyperparameters = OrderedDict()
            for column in trials_data.columns:
                best_hyperparameters[column] = convert_numpy(trials_data[column][index])
            best_hyperparameters = dict(best_hyperparameters)
            with open(f'{algorithm}_{scenario}_best_hyperparameters.yaml', 'w') as file:
                yaml.dump(best_hyperparameters, file, default_flow_style=False, sort_keys=False)

    # Convert to DataFrame for plotting
    df = pd.DataFrame(data, columns=['Scenario', 'Hyperparmeter', 'Normalized Tracking Reward', 'Normalized Action Change Reward'])

    rmse_palette = sns.color_palette("viridis", len(scenarios) + 1).as_hex()
    rms_palette = sns.color_palette("coolwarm", len(scenarios) + 1).as_hex()

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Scenario', y='Normalized Tracking Reward', hue='Hyperparmeter', data=df[['Scenario', 'Hyperparmeter', 'Normalized Tracking Reward']], palette=rmse_palette)
    sns.move_legend(plt.gca(), 'lower left')
    plt.title('Performance Comparison of Hand-Tuned and Optimized Hyperparameters')
    plt.savefig(f'hpo_performance_comparison_in_different_scenarios_{algorithm}.png')

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Scenario', y='Normalized Action Change Reward', hue='Hyperparmeter', data=df[['Scenario', 'Hyperparmeter', 'Normalized Action Change Reward']])
    sns.move_legend(plt.gca(), 'lower left')
    plt.title('Action Change Comparison of Hand-Tuned and Optimized Hyperparameters')
    plt.savefig(f'hpo_action_change_comparison_in_different_scenarios_{algorithm}.png')

    plt.figure(figsize=(14, 8))
    # Aggregate data for stacking
    tracking_data = df.groupby(["Scenario", "Hyperparmeter"])["Normalized Tracking Reward"].sum().unstack()
    action_data = df.groupby(["Scenario", "Hyperparmeter"])["Normalized Action Change Reward"].sum().unstack()
    # Bar positions
    scenarios = tracking_data.index
    x = np.arange(len(scenarios))  # Numerical positions for algorithms
    bar_width = 0.4  # Width of each bar group

    # Plot hand-tuned data
    plt.bar(x - 0.5*bar_width, -np.log(tracking_data["handtuned"]), label="RMSE (Hand Tuned)", color=rmse_palette[0], width=bar_width)
    plt.bar(x - 0.5*bar_width, -np.log(action_data["handtuned"]), bottom=-np.log(tracking_data["handtuned"]), label="RMS Action Change (Hand Tuned)", color=rms_palette[0], width=bar_width)

    # Plot optimized (stacked)
    plt.bar(x + 0.5*bar_width, -np.log(tracking_data["optimized"]), label="RMSE (Optimized)", color=rmse_palette[1], width=bar_width)
    plt.bar(x + 0.5*bar_width, -np.log(action_data["optimized"]), bottom=-np.log(tracking_data["optimized"]), label="RMS Action Change (Optimized)", color=rms_palette[1], width=bar_width)

    # Add labels, legend, and title
    plt.xticks(x, scenarios)  # Rotate x-axis labels for clarity
    plt.xlabel("Scenario")
    plt.ylabel("RMSE and RMS Action Change")
    plt.title("RMSE and RMS Action Change Comparison of Hand-Tuned and Optimized Hyperparameters")
    plt.legend()
    plt.savefig(f"grouped_stacked_hpo_rms_chart_{algorithm}.png")

    plt.figure(figsize=(14, 8))
    # Plot hand-tuned data
    plt.bar(x - 0.5*bar_width, tracking_data["handtuned"], label="Tracking Reward (Hand Tuned)", color=rmse_palette[0], width=bar_width)
    plt.bar(x - 0.5*bar_width, action_data["handtuned"], bottom=tracking_data["handtuned"], label="Action Reward (Hand Tuned)", color=rms_palette[0], width=bar_width)

    # Plot optimized (stacked)
    plt.bar(x + 0.5*bar_width, tracking_data["optimized"], label="Tracking Reward (Optimized)", color=rmse_palette[1], width=bar_width)
    plt.bar(x + 0.5*bar_width, action_data["optimized"], bottom=tracking_data["optimized"], label="Action Reward (Optimized)", color=rms_palette[1], width=bar_width)

    # Add labels, legend, and title
    plt.xticks(x, scenarios)  # Rotate x-axis labels for clarity
    plt.xlabel("Scenario")
    plt.ylabel("Normalized Rewards")
    plt.title("Performance Comparison of Hand-Tuned and Optimized Hyperparameters")
    plt.legend(loc='lower left')
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"grouped_stacked_hpo_chart_{algorithm}.png")

# # Plot evaluation over trials for each algorithm
# plot_hpo_evaluation(trials, algorithms)

# # Plot box plot comparison across algorithms
# plot_performance_comparison(algorithms)

# Plot Pareto front analysis for multi-objective optimization
print("Generating Pareto front analysis...")
print("Pareto front shows the trade-off between RMSE (tracking performance) and RMS Action Change (control smoothness)")
print("Points on the Pareto front represent optimal solutions where improving one objective requires compromising the other")
print(f"HPO Selection Criterion: {metric_weights[0]:.2f} × RMSE + {metric_weights[1]:.2f} × RMS_Action_Change")
print("Star markers (*) indicate the points selected by HPO based on this weighted combination")

# Check what algorithms and packages actually have data
available_algorithms = []
available_packages = []

for algorithm in algorithms:
    try:
        # Check basic scenario for vizier package
        trials_data = load_trials_data(algorithm, 'basic', base_dir)
        if trials_data is not None:
            available_algorithms.append(algorithm)
            if 'basic' not in available_packages:
                available_packages.append('basic')
    except:
        print(f"No data found for {algorithm}")

print(f"Available algorithms: {available_algorithms}")
print(f"Available packages/scenarios: {available_packages}")

if available_algorithms and available_packages:
    plot_pareto_front(available_algorithms, available_packages)
    plot_pareto_front_by_algorithm(available_algorithms, available_packages)
else:
    print("No HPO data available for Pareto front analysis")

# Plot box plot comparison across scenarios for each algorithm
for algorithm in algorithms:
    plot_performance_comparison_in_different_scenarios(algorithm, scenarios)
