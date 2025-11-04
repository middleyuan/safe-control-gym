import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# 1. --- Data Setup ---
# Define the categories and the final values for the plot
categories = ['Combat', 'Durability', 'Intelligence', 'Power', 'Speed', 'Strength']
final_values = np.array([85, 90, 75, 95, 80, 88])  # Use a NumPy array for easier calculations

# Animation settings
num_animation_frames = 60  # More frames for a smoother animation

# 2. --- Plotting Setup ---
# Calculate the angle for each axis.
num_categories = len(categories)
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
# The plot must be closed, so we repeat the first angle at the end.
angles += angles[:1]

# Create the figure and polar subplot
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Set category labels and radial axis limits
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 100)
ax.set_rlabel_position(30)  # Move radial labels to avoid overlap
ax.grid(color='grey', linestyle='--', linewidth=0.5)

# 3. --- Animation ---
# To make the plot 'grow', we start with a line at the center (all values are 0).
# The first value is repeated at the end to close the plot.
initial_values_closed = np.zeros(len(final_values) + 1)
line, = ax.plot(angles, initial_values_closed, 'o-', linewidth=2.5, color='dodgerblue')
fill = ax.fill(angles, initial_values_closed, alpha=0.2, color='dodgerblue')[0]


# Animation function: this is called for each frame
def update(frame):
    '''Updates the plot for a specific frame, making the plot grow.

    Args:
        frame (int): The current frame number, from 0 to num_animation_frames-1.
    '''
    # Calculate the progress of the animation (from 0.0 to 1.0)
    progress = frame / (num_animation_frames - 1)

    # Interpolate the values from 0 to their final state
    current_values = final_values * progress

    # Close the loop for the plot
    current_values_closed = np.concatenate((current_values, [current_values[0]]))

    # Update the data for the line and the filled area
    line.set_ydata(current_values_closed)

    # To update the fill, we need to reset its path
    # Path is a list of (x, y) vertices
    path = np.column_stack([angles, current_values_closed])
    fill.set_xy(path)

    return line, fill


# Create the animation object
# interval: Delay between frames in milliseconds. Smaller is faster.
ani = FuncAnimation(fig, update, frames=num_animation_frames, interval=25, blit=True)

# To save the animation as a GIF (requires 'pillow' writer)
# pip install pillow
# ani.save('growing_spider_plot.gif', writer='pillow', fps=30)

# Display the animation
ani.save('growing_spider_plot.gif', writer='pillow', fps=30)
# plt.show()
