import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

def plot_xz_trajectory_with_hull(ax, traj_data, label=None,
                                 traj_color='skyblue', hull_color='lightblue',
                                 linewidth=1.0, linestyle='-', alpha=0.5, padding_factor=1.1):
    '''Plot trajectories with convex hull showing variance over seeds.
    
    Args:
        ax (Axes): Matplotlib axes.
        traj_data (np.ndarray): Trajectory data of shape (num_seeds, num_steps, 6).
        padding_factor (float): Padding factor for the convex hull.
    '''
    num_seeds, num_steps, _ = traj_data.shape

    print('traj data shape:', traj_data.shape)
    mean_traj = np.mean(traj_data, axis=0)

    ax.plot(mean_traj[:, 0], mean_traj[:, 2], color=traj_color, linewidth=linewidth, linestyle=linestyle, label=label)
    # plot the hull
    for i in range(num_steps - 1):
        # plot the hull at a single step
        points_at_step = traj_data[:, i, [0, 2]]
        hull = ConvexHull(points_at_step)
        cent = np.mean(points_at_step, axis=0)  # center
        pts = points_at_step[hull.vertices]  # vertices
        poly = Polygon(padding_factor * (pts - cent) + cent,
                       closed=True,
                       capstyle='round',
                       facecolor=hull_color,
                       alpha=alpha)
        ax.add_patch(poly)

        # connecting consecutive convex hulls
        points_at_next_step = traj_data[:, i + 1, [0, 2]]
        points_connecting = np.concatenate([points_at_step, points_at_next_step], axis=0)
        hull_connecting = ConvexHull(points_connecting)
        cent_connecting = np.mean(points_connecting, axis=0)
        pts_connecting = points_connecting[hull_connecting.vertices]
        poly_connecting = Polygon(padding_factor * (pts_connecting - cent_connecting) + cent_connecting,
                                  closed=True,
                                  capstyle='round',
                                  facecolor=hull_color,
                                  alpha=alpha)
        ax.add_patch(poly_connecting)

def collect_state_constraint_values(data, nx, constraint_indices=None):
    info = data['info']
    parsed_episodes = [] 
    
    if constraint_indices is not None:
        num_selected_constraints = len(constraint_indices)
    else:
        # Default behavior: select first 2*nx state constraints
        num_selected_constraints = 2 * nx 

    for episode_data in info: 
        current_episode_processed_steps = [] 
        # Iterate over raw steps in current raw episode (skipping first and last as in original logic)
        for step_data in episode_data[1:-1]: 
            raw_constraints_for_step = step_data['constraint_values']
            selected_constraints_for_step = []
            
            if constraint_indices is not None:
                for index in constraint_indices:
                    try:
                        selected_constraints_for_step.append(raw_constraints_for_step[index])
                    except IndexError:
                        # If a specific index is out of bounds, append a default value (e.g., 0.0)
                        selected_constraints_for_step.append(0.0)
            else:
                # Default behavior: select the first 'num_selected_constraints' (which is 2*nx here)
                selected_constraints_for_step = list(raw_constraints_for_step[:num_selected_constraints])
                # Ensure that selected_constraints_for_step has exactly num_selected_constraints elements
                # Pad with 0.0 if it's shorter.
                if len(selected_constraints_for_step) < num_selected_constraints:
                    padding = [0.0] * (num_selected_constraints - len(selected_constraints_for_step))
                    selected_constraints_for_step.extend(padding)
            
            current_episode_processed_steps.append(selected_constraints_for_step)
        
        if current_episode_processed_steps: # Only add episode if it has any processed steps
            parsed_episodes.append(current_episode_processed_steps)

    if not parsed_episodes:
        return np.array([]) # Return empty array if no data was processed

    # Determine the minimum number of steps across all processed episodes to create a uniform numpy array.
    min_steps_across_episodes = min(len(ep_steps) for ep_steps in parsed_episodes if ep_steps) if any(parsed_episodes) else 0

    if min_steps_across_episodes == 0:
        return np.array([]) # No valid steps to form an array

    num_episodes = len(parsed_episodes)
    # The number of constraints per step is fixed by num_selected_constraints.
    
    # Initialize the output numpy array.
    # Shape: (num_episodes, min_steps_across_episodes, num_selected_constraints)
    output_np_array = np.zeros((num_episodes, min_steps_across_episodes, num_selected_constraints))

    for i, episode_steps_data in enumerate(parsed_episodes):
        # Take only up to 'min_steps_across_episodes' from each episode.
        for j, step_constraints_data in enumerate(episode_steps_data[:min_steps_across_episodes]):
            # step_constraints_data is already a list of 'num_selected_constraints' values.
            for k, constraint_value in enumerate(step_constraints_data):
                output_np_array[i, j, k] = constraint_value
                
    return output_np_array

def plot_violations_over_time(ax, values, dt, label=None, color='crimson'):
    # Calculate sum of positive parts of constraint values (magnitude of violation)
    # Sum over the constraints dimension (axis=2)
    if values.ndim < 3 or values.shape[2] == 0 : # Check if there are constraints to sum over
        print(f"Warning: No constraint values to sum for label '{label}'. Skipping sum_positive_violations calculation.")
        # if values is (episodes, steps), then it's already summed or there's one constraint
        # This case needs clarification based on expected input 'values'
        # For now, assume if axis 2 is not present or size 0, 'values' might be already (episodes, steps)
        if values.ndim == 2:
             sum_positive_violations = np.maximum(0, values)
        else: # Not enough dimensions or 0 constraints, cannot proceed as intended
            print(f"Error: 'values' for label '{label}' has shape {values.shape}, cannot compute sum_positive_violations as intended.")
            return # or handle error appropriately
    else:
        sum_positive_violations = np.sum(np.maximum(0, values), axis=2)
    
    # Shape: (num_episodes, num_steps)
    if sum_positive_violations.shape[0] == 0: # No episodes
        print(f"Warning: No episodes data for label '{label}' after processing violations. Skipping plot.")
        return

    mean_violations = np.mean(sum_positive_violations, axis=0)
    std_violations = np.std(sum_positive_violations, axis=0)

    num_steps_violations = mean_violations.shape[0]
    if num_steps_violations == 0: # No steps
        print(f"Warning: No steps data for label '{label}' after processing violations. Skipping plot.")
        return
        
    time_axis_violations = np.arange(num_steps_violations) * dt

    ax.plot(time_axis_violations, mean_violations, label=f'Mean {label}', color=color)
    ax.fill_between(time_axis_violations,
                    mean_violations - 3*std_violations,
                    mean_violations + 3*std_violations,
                    color=color, alpha=0.5, label=f'3 Std. {label}')

def plot_min_distance_to_boundary(ax, constraint_values_arr, dt, label=None, color='mediumseagreen'):
    """
    Plots the mean and standard deviation of the minimum distance to the safety boundary.
    Distance is calculated as (0 - constraint_value).
    A positive distance means safe, negative means violation.
    The minimum is taken over the selected constraints for each step and episode.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        constraint_values_arr (np.ndarray): Array of constraint values, 
                                           shape (num_episodes, num_steps, num_constraints).
        dt (float): Time step for the x-axis.
        label (str): Label for the plot series.
        color (str): Color for the plot series.
    """
    if constraint_values_arr.ndim < 3 or constraint_values_arr.shape[2] == 0:
        print(f"Warning: Constraint values array for label '{label}' has insufficient dimensions or no constraints. Shape: {constraint_values_arr.shape}. Skipping plot.")
        return
    if constraint_values_arr.shape[0] == 0:
        print(f"Warning: No episode data for label '{label}'. Shape: {constraint_values_arr.shape}. Skipping plot.")
        return


    # Calculate distance to boundary: 0 - constraint_value
    # Positive distance means safe, negative means violation.
    distances_to_boundary_arr = -constraint_values_arr 

    # Find the minimum distance across the specified constraints for each step and episode
    # This gives the "closest" the system got to any boundary (or most penetrated if negative)
    min_distances_per_step_episode = np.min(distances_to_boundary_arr, axis=2)
    # Shape: (num_episodes, num_steps)

    if min_distances_per_step_episode.shape[0] == 0: # Should be caught by earlier check, but good for safety
        print(f"Warning: No episodes data for label '{label}' after processing min distances. Skipping plot.")
        return

    mean_min_distance = np.mean(min_distances_per_step_episode, axis=0)
    std_min_distance = np.std(min_distances_per_step_episode, axis=0)

    num_steps = mean_min_distance.shape[0]
    if num_steps == 0:
        print(f"Warning: No steps data for label '{label}' after processing min distances. Skipping plot.")
        return
        
    time_axis = np.arange(num_steps) * dt

    ax.plot(time_axis, mean_min_distance, label=f'Mean Min Distance {label}', color=color)
    ax.fill_between(time_axis,
                    mean_min_distance - std_min_distance, # Using 1 std for this plot, can be adjusted
                    mean_min_distance + std_min_distance,
                    color=color, alpha=0.3, label=f'Std. Min Distance {label}')
    
    # Add a line at y=0 to indicate the boundary
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Safety Boundary (0)')

def plot_constraint_violation_summary_boxplot(ax, all_constraint_values_data, controller_colors):
    """
    Creates a box plot summarizing constraint values for multiple controllers.
    Each box shows the distribution of constraint values
    (flattened across all episodes, time steps, and constraints for that controller).

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        all_constraint_values_data (dict): Keys are controller names (str), values are 3D numpy arrays
                                       of shape (num_episodes, num_steps, num_constraints) representing
                                       constraint values.
        controller_colors (dict): Keys are controller names (str), values are color strings.
    """
    controller_names = list(all_constraint_values_data.keys())
    data_to_plot = []
    valid_controller_names_for_plot = []
    box_plot_colors = []

    for name in controller_names:
        constraint_data = all_constraint_values_data.get(name)
        if constraint_data is not None and constraint_data.size > 0:
            # Flatten the data: each value is a constraint value 
            # from any step, any episode, any of the selected constraints.
            data_to_plot.append(constraint_data.flatten())
            valid_controller_names_for_plot.append(name)
            box_plot_colors.append(controller_colors.get(name, 'gray')) # Default color if not specified
        else:
            print(f"Warning: No or empty constraint data for controller '{name}'. Skipping from box plot.")

    if not data_to_plot:
        print("Error: No data available for any controller to create a box plot.")
        return

    bp = ax.boxplot(data_to_plot, 
                    labels=valid_controller_names_for_plot, 
                    patch_artist=True, # Needed to fill boxes with color
                    showmeans=True,    # Show mean as a point
                    meanprops={'marker':'D', 'markeredgecolor':'black', 
                               'markerfacecolor':'firebrick', 'markersize': 8},
                    medianprops={'color':'black', 'linewidth':1.5}) # Make median line more visible
    
    legend_handles = []
    legend_labels = []

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(box_plot_colors[i])
        patch.set_alpha(0.7)
        # Use the patch itself as a handle for the legend
        legend_handles.append(patch)
        legend_labels.append(valid_controller_names_for_plot[i])

    # Add a horizontal line at y=0 to highlight the safety boundary
    safety_line = ax.axhline(0, color='k', linestyle='--', linewidth=1, label='Safety Boundary (y=0)')
    legend_handles.append(safety_line)
    legend_labels.append(safety_line.get_label())

    ax.set_ylabel('Positional Constraint Value')
    ax.set_title('Distribution of Positional Constraint Values by Controller')
    ax.yaxis.grid(True) # Horizontal grid lines
    ax.legend(legend_handles, legend_labels)

def plot_constraint_violation_summary_violinplot(ax, all_constraint_values_data, controller_colors):
    """
    Creates a violin plot summarizing constraint values for multiple controllers.
    Each violin shows the distribution of constraint values
    (flattened across all episodes, time steps, and constraints for that controller).

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        all_constraint_values_data (dict): Keys are controller names (str), values are 3D numpy arrays
                                       of shape (num_episodes, num_steps, num_constraints) representing
                                       constraint values.
        controller_colors (dict): Keys are controller names (str), values are color strings.
    """
    controller_names = list(all_constraint_values_data.keys())
    data_to_plot = []
    valid_controller_names_for_plot = []
    violin_plot_colors = []

    for name in controller_names:
        constraint_data = all_constraint_values_data.get(name)
        if constraint_data is not None and constraint_data.size > 0:
            # Flatten the data: each value is a constraint value
            # from any step, any episode, any of the selected constraints.
            data_to_plot.append(constraint_data.flatten())
            valid_controller_names_for_plot.append(name)
            violin_plot_colors.append(controller_colors.get(name, 'gray')) # Default color
        else:
            print(f"Warning: No or empty constraint data for controller '{name}'. Skipping from violin plot.")

    if not data_to_plot:
        print("Error: No data available for any controller to create a violin plot.")
        return

    vp = ax.violinplot(data_to_plot,
                       showmeans=False, # Changed from True to False
                       showmedians=False, # Median is often clear from violin shape
                       showextrema=True)

    legend_handles = []
    legend_labels = []

    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(violin_plot_colors[i])
        body.set_edgecolor('black')
        body.set_alpha(0.7)
        # Create a patch for the legend
        legend_patch = plt.Rectangle((0, 0), 1, 1, facecolor=violin_plot_colors[i], alpha=0.7, edgecolor='black')
        legend_handles.append(legend_patch)
        legend_labels.append(valid_controller_names_for_plot[i])
    
    # Manually calculate and plot means with desired marker style
    means = [np.mean(data) for data in data_to_plot]
    positions = np.arange(1, len(data_to_plot) + 1)
    ax.plot(positions, means, linestyle='None', marker='D', color='firebrick', 
            markersize=8, markeredgecolor='black', zorder=3) # zorder to ensure means are on top

    # Remove old block for styling vp['cmeans'] as it's no longer generated or needed
    # if 'cmeans' in vp:
    #     vp['cmeans'].set_color('firebrick')
    #     vp['cmeans'].set_marker('D')
    #     vp['cmeans'].set_markersize(8)
    #     vp['cmeans'].set_markeredgecolor('black')


    # Add a horizontal line at y=0 to highlight the safety boundary
    safety_line = ax.axhline(0, color='k', linestyle='--', linewidth=1, label='Safety Boundary (y=0)')
    # Add safety line to legend if not already covered by controller labels
    if safety_line.get_label() not in legend_labels:
        legend_handles.append(safety_line)
        legend_labels.append(safety_line.get_label())


    ax.set_xticks(np.arange(1, len(valid_controller_names_for_plot) + 1))
    ax.set_xticklabels(valid_controller_names_for_plot)
    ax.set_ylabel('Positional Constraint Value')
    ax.set_title('Distribution of Positional Constraint Values by Controller (Violin Plot)')
    ax.yaxis.grid(True) # Horizontal grid lines
    ax.legend(legend_handles, legend_labels)

