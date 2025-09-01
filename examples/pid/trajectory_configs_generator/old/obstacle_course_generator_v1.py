import yaml
from dataclasses import dataclass
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
# Custom Dumper: Force inline lists like [x, y, z], but multiline dicts
class InlineListDumper(yaml.SafeDumper):
    pass

def represent_inline_list(dumper, data):
    # Force inline format for 3-element numeric lists (e.g. [x, y, z])
    if all(isinstance(i, (int, float)) for i in data) and len(data) == 3:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data)

InlineListDumper.add_representer(list, represent_inline_list)

@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    t: float
    label: str = ""  # inject section label

    def to_dict(self):
        return {
            "time": round(self.t, 2),
            "position": [round(self.x, 3), round(self.y, 3), round(self.z, 3)]
        }


@dataclass
class StringObstacle:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    label: str = ""  # inject section label

    def to_dict(self):
        return {
            "start": [round(v, 3) for v in self.start],
            "end": [round(v, 3) for v in self.end]
        }


class TrajectoryBuilder:
    def __init__(self):
        self.waypoints: List[Waypoint] = []
        self.strings: List[StringObstacle] = []

    def add_waypoints(self, wps: List[Waypoint]):
        self.waypoints.extend(wps)

    def add_strings(self, strings: List[StringObstacle]):
        self.strings.extend(strings)

    def order_waypoints_by_time(self):
        self.waypoints.sort(key=lambda wp: wp.t)
    
    def generate_grid_waypoints(self, xs, ys, zs, time_step=1.0, start_time=0.0) -> List[Waypoint]:
        wps = []
        t = start_time
        for x in xs:
            for y in ys:
                for z in zs:
                    wps.append(Waypoint(x, y, z, t))
                    t += time_step
        return wps
    
    def generate_time_waypoints(self, wps: List[Tuple[float, float, float]], time_step=1.0, start_time=0.0) -> List[Waypoint]:
        waypoints = []
        t = start_time
        for x, y, z in wps:
            waypoints.append(Waypoint(x, y, z, t))
            t += time_step
        return waypoints
    
    def generate_time_fixed_waypoints(self, wps: List[Tuple[float, float, float]], fwps, fix_axis, time_step=1.0, start_time=0.0) -> List[Waypoint]:
        waypoints = []
        t = start_time
        for x, y, z in wps:
            if fix_axis == 'x':
                waypoints.append(Waypoint(fwps, y, z, t))
            elif fix_axis == 'y':
                waypoints.append(Waypoint(x, fwps, z, t))
            elif fix_axis == 'z':
                waypoints.append(Waypoint(x, y, fwps, t))
            t += time_step
        return waypoints
    
    def add_waypoints_with_dt(self, wps: List[Tuple[float, float, float, float]], start_time: float = 0.0):
        """
        Add waypoints where each waypoint's time is the previous time plus dt from wps.
        wps: list of (x, y, z, dt) tuples
        start_time: initial time for the first waypoint
        """
        t = start_time
        for x, y, z, dt in wps:
            self.waypoints.append(Waypoint(x, y, z, t))
            t += dt

    def generate_line_strings(self, xs, ys, zs, axis='z', length = 1) -> List[StringObstacle]:
        strings = []
        for x in xs:
            for y in ys:
                for z in zs:
                    if axis == 'z':
                        strings.append(StringObstacle((x, y, z-length/2), (x, y, z+length/2)))
                    elif axis == 'y':
                        strings.append(StringObstacle((x, y-length/2, z), (x, y+length/2, z)))
                    elif axis == 'x':
                        strings.append(StringObstacle((x-length/2, y, z), (x+length/2, y, z)))
        return strings

    def concatenate(self, other: 'TrajectoryBuilder', time_offset: float = 0.0, label: str = ""):
        for wp in other.waypoints:
            self.waypoints.append(Waypoint(wp.x, wp.y, wp.z, wp.t + time_offset, label=label))
        for s in other.strings:
            self.strings.append(StringObstacle(s.start, s.end, label=label))
    def add_vertical_strings(self, xs: List[float], ys: List[float], z_min: float = 0.0, z_max: float = 1.8, label: str = ""):
        """
        Add vertical string obstacles at each (x, y) coordinate, spanning from z_min to z_max.
        """
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i] if i < len(ys) else 0.0
            self.strings.append(
                    StringObstacle(start=(x, y, z_min), end=(x, y, z_max), label=label)
            )

    def to_yaml_config(self) -> str:
        def represent_ordered_dict(dumper, data):
            return dumper.represent_dict(data.items())

        yaml.add_representer(dict, represent_ordered_dict, Dumper=InlineListDumper)

        # Manually build the config dict with placeholders for strings/waypoints
        base_config = {
            "task_config": {
                "info_in_reset": True,
                "ctrl_freq": 60,
                "pyb_freq": 60,
                "physics": "dyn_si_3d_delay",
                "quad_type": 9,
                "init_state": {
                    # "init_x": 1.5,
                    # "init_x_dot": 0,
                    # "init_y": 1.5,
                    # "init_y_dot": 0,
                    "init_x": 2.75,
                    "init_x_dot": 0,
                    "init_y": 1.5,
                    "init_y_dot": 0,
                    "init_z": 1.2,
                    "init_z_dot": 0,
                },
                "task": "traj_tracking",
                "task_info": {
                    "trajectory_type": "snap_custom",
                    "num_cycles": 1,
                    "trajectory_plane": "xy",
                    "trajectory_position_offset": [0.0, 0.0, 0.0],
                    "trajectory_scale": 1.0,
                    "strings": "__STRINGS__",
                    "waypoints": "__WAYPOINTS__",
                },
                "inertial_prop": {"M": 0.037},
                "episode_len_sec": float(self.waypoints[-1].t) if self.waypoints else 20,
                "cost": "quadratic",
                "constraints": [
                    {"constraint_form": "default_constraint", "constrained_variable": "state"},
                    {"constraint_form": "default_constraint", "constrained_variable": "input"}
                ],
                "done_on_violation": False
            }
        }

        # Dump the config excluding waypoints/strings
        config_text = yaml.dump(base_config, sort_keys=False, Dumper=InlineListDumper)

        # Group by label
        def group_with_labels(items):
            grouped = {}
            for item in items:
                grouped.setdefault(item.label or "Unlabeled", []).append(item)
            return grouped

        def format_list_with_comments(grouped_dict, key_name):
            lines = [f"    {key_name}:"]
            for label, items in grouped_dict.items():
                lines.append(f"      # === {label} ===")
                for item in items:
                    item_yaml = yaml.dump(item.to_dict(), Dumper=InlineListDumper).strip().splitlines()
                    lines.append(f"      - {item_yaml[0]}")
                    for line in item_yaml[1:]:
                        lines.append(f"        {line}")
            return lines

        string_lines = format_list_with_comments(group_with_labels(self.strings), "strings")
        waypoint_lines = format_list_with_comments(group_with_labels(self.waypoints), "waypoints")

        # Replace placeholders
        config_lines = config_text.splitlines()
        final_lines = []
        for line in config_lines:
            if line.strip() == "strings: __STRINGS__":
                final_lines.extend(string_lines)
            elif line.strip() == "waypoints: __WAYPOINTS__":
                final_lines.extend(waypoint_lines)
            else:
                final_lines.append(line)

        return "\n".join(final_lines)
    def apply_shrinking_factor(self, shrink_factor: float, preserve_z: bool = True):
        """
        Shrink all waypoints and strings towards (0, 0) by the given factor.
        
        Args:
            shrink_factor: Factor to shrink by (e.g., 0.8 = 80% of original size)
            preserve_z: If True, only shrink x and y coordinates, keep z unchanged
        """
        # Shrink waypoints
        for wp in self.waypoints:
            wp.x *= shrink_factor
            wp.y *= shrink_factor
            if not preserve_z:
                wp.z *= shrink_factor
        
        # Shrink strings
        for string in self.strings:
            # Shrink start point
            start_list = list(string.start)
            start_list[0] *= shrink_factor  # x
            start_list[1] *= shrink_factor  # y
            if not preserve_z:
                start_list[2] *= shrink_factor  # z
            string.start = tuple(start_list)
            
            # Shrink end point
            end_list = list(string.end)
            end_list[0] *= shrink_factor  # x
            end_list[1] *= shrink_factor  # y
            if not preserve_z:
                end_list[2] *= shrink_factor  # z
            string.end = tuple(end_list)

    def apply_translation(self, translation: Tuple[float, float, float]):
        """
        Translate all waypoints and strings by the given offset.
        
        Args:
            translation: (dx, dy, dz) translation offset
        """
        dx, dy, dz = translation
        
        # Translate waypoints
        for wp in self.waypoints:
            wp.x += dx
            wp.y += dy
            wp.z += dz
        
        # Translate strings
        for string in self.strings:
            # Translate start point
            start_list = list(string.start)
            start_list[0] += dx
            start_list[1] += dy
            start_list[2] += dz
            string.start = tuple(start_list)
            
            # Translate end point
            end_list = list(string.end)
            end_list[0] += dx
            end_list[1] += dy
            end_list[2] += dz
            string.end = tuple(end_list)

    def transform_trajectory(self, shrink_factor: float = 1.0, translation: Tuple[float, float, float] = (0.0, 0.0, 0.0), preserve_z_shrinking: bool = True):
        """
        Apply shrinking and translation in one step.
        
        Args:
            shrink_factor: Factor to shrink towards (0,0) - applied first
            translation: (dx, dy, dz) offset - applied second
            preserve_z_shrinking: If True, only shrink x,y coordinates
        """
        if shrink_factor != 1.0:
            self.apply_shrinking_factor(shrink_factor, preserve_z_shrinking)
        
        if translation != (0.0, 0.0, 0.0):
            self.apply_translation(translation)

def plot_trajectory_3d(builder: TrajectoryBuilder, show=True, save_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot waypoints with color gradient and numbers
    xs = [wp.x for wp in builder.waypoints]
    ys = [wp.y for wp in builder.waypoints]
    zs = [wp.z for wp in builder.waypoints]
    num_wps = len(builder.waypoints)
    colors = plt.cm.viridis(np.linspace(0, 1, num_wps))
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        ax.scatter(x, y, z, color=colors[i], s=60)
        ax.text(x, y, z, f"{i+1}", color='black', fontsize=10, ha='center', va='center')

    ax.plot(xs, ys, zs, color='gray', linestyle='--', label='Trajectory')

    # Plot strings
    for s in builder.strings:
        x = [s.start[0], s.end[0]]
        y = [s.start[1], s.end[1]]
        z = [s.start[2], s.end[2]]
        ax.plot(x, y, z, color='red', linewidth=2, label='String' if 'String' not in ax.get_legend_handles_labels()[1] else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory and String Obstacles')
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1,1,0.7])

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

   
if __name__ == "__main__":
      # obstacle course should be within
        # x: [-3, 3]
        # y: [-2, 2]
        # z: [0, 2.5]
    set_time_step = 1.5
    # set_time_step = 1.0
    # set_time_step = 0.8
    # fixed_z = 1.0
    fixed_z = 0.8

    # Transformation parameters
    shrink_factor = 0.9  # Shrink to 80% of original size
    translation = (0.0, 0.0, 0.0)  # Translate by (0.0, 0.0, 0.0)
    preserve_z_shrinking = True  # Only shrink x,y, preserve z heights

    # Section 1: Randomized obstacles + waypoints
    section1 = TrajectoryBuilder()
    # Add waypoints & obstacles for section 1
    section1.add_waypoints_with_dt([
        (2.75, 1.5, 1.2, set_time_step), # 1
        (1.75, 1.85, 1.35, set_time_step*1.5), # 2
        (1.1, 1.0, 1.5, set_time_step), # 3
        (2.0, 0.0, 1.2, set_time_step), # 4
        (2.75, -1.25, 1.0, set_time_step*1.25), # 5
        (2.0, -1.75, 0.8, set_time_step*1.25), # 6
        (0.4, -0.8, 0.7, set_time_step*1.25), # 7
        (1.6, 1.1, 0.8, set_time_step), # 8
        (1.0, 1.5, 1.0, set_time_step*1.25), # 9
        (0.5, 0.9, 1.2, set_time_step*1.25), # 10
        (1.5, -1.1, 1.3, set_time_step*1.5), # 11
    ], start_time=0.0)

    section1.add_strings([
        StringObstacle((2.0, 1.25, 1.8), 
                       (0.0, 0.75, 0.0)),
        StringObstacle((2.0, -1.25, 0.0), 
                       (0.0, -0.75, 1.8)),
        StringObstacle((2.0, -1.25, 1.8), 
                       (2.0, 1.25, 1.8)),
        ])
    section1.add_vertical_strings(
        [0.0, 0.0, 2.0, 2.0],
        [-0.75, 0.75, -1.25, 1.25]
    )
    section1.order_waypoints_by_time()
    
    
    # Section 2: Randomized obstacles + waypoints
    section2 = TrajectoryBuilder()
    factor = 0.75
    factor2 = 0.5
    # Add waypoints & obstacles for section 2
    section2.add_waypoints_with_dt([
        (0.5, -1.5, 1.2, set_time_step*factor), # 12
        (-0.5, -1.25, 1.2, set_time_step*factor*factor2), # 13
        (-1.8, -0.6, 1.35, set_time_step*factor*factor2), # 14
        (-2.75, -1.25, 1.5, set_time_step*factor), # 15
        (-2.0, -1.75, 1.35, set_time_step*factor), # 16
        (-1.25, -1.25, 1.2, set_time_step*factor), # 17
        (-0.5, -0.3, 1.0, set_time_step*factor), # 18
        (-0.75, 0.6, 0.8, set_time_step*factor), # 19
        (-1.5, 0.75, 0.6, set_time_step*factor), # 20
        (-2.0, 0.6, 0.8, set_time_step*factor*factor2), # 21
        (-2.5, 0.25, 1.2, set_time_step*factor*factor2), # 22
        (-2.0, -0.25, 1.35, set_time_step*factor), # 23
        (-0.5, -0.25, 1.5, set_time_step*factor),  # 24
        (0.5, 0.9, 1.5, set_time_step*factor), # 25
        (0.0, 1.5, 1.2, set_time_step*factor), # 26
    ], start_time=0.0)

    section2.add_strings([
        StringObstacle((0.0, 0.75, 1.8), 
                       (-2.0, -1.25, 1.8)),
        StringObstacle((0.0, -0.75, 0.0), 
                       (-2.0, 1.25, 1.2)),
        StringObstacle((-2.0, 1.25, 1.8), 
                       (-2.0, -1.25, 0.0)),
        ])
    
    section2.add_vertical_strings(
        [-2.0, -2.0],
        [-1.25, 1.25]
    )

    section2.order_waypoints_by_time()
    # Section 3: Speed Test
    section3 = TrajectoryBuilder()
    # Add waypoints & obstacles for section 3
    fixed_z = 1.0
    factor = 0.6
    factor2 = 0.75
    section3.add_waypoints_with_dt([
        (-1.0, 2.0, fixed_z, set_time_step*factor), # 27
        (-2.5, 1.5, fixed_z, set_time_step*factor), # 28
        (-2.75, 0.5, fixed_z, set_time_step*factor), # 29
        (-2.75, -0.5, fixed_z, set_time_step*factor), # 30
        (-2.5, -1.5, fixed_z, set_time_step*factor), # 31
        (-1.0, -2.0, fixed_z, set_time_step*factor), # 32
        (1.0, -2.0, fixed_z, set_time_step*factor), # 33
        (2.5, -1.5, fixed_z, set_time_step*factor), # 34
        (2.75, -0.5, fixed_z, set_time_step*factor2), # 35
        (2.75, 0.5, fixed_z, set_time_step*factor2), # 36
        (2.75, 1.5, fixed_z, set_time_step*factor2), # 37
    ], start_time=0.0)

    final_trajectory = TrajectoryBuilder()
    
    # Section 1
    final_trajectory.concatenate(section1, time_offset=0, label="Section: Randomized String Obstacles")
    offset_1 = section1.waypoints[-1].t + set_time_step
    
    # Section 2
    final_trajectory.concatenate(section2, time_offset=offset_1, label="Section 2: Fixed Strings")
    offset_2 = offset_1 + section2.waypoints[-1].t + set_time_step  # accumulate correctly

    # Section 3
    final_trajectory.concatenate(section3, time_offset=offset_2, label="Section 3: Disturbance Obstacles")
    final_trajectory.order_waypoints_by_time()

    # Apply transformations
    final_trajectory.transform_trajectory(
        shrink_factor=shrink_factor,
        translation=translation,
        preserve_z_shrinking=preserve_z_shrinking
    )

    # Export to YAML config
    config_str = final_trajectory.to_yaml_config()

    file_name = f"final_trajectory_prototype_{float(set_time_step)}.yaml"
    
    # Ensure the output directory exists
    output_dir = "trajectory_configs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    plot_trajectory_3d(final_trajectory, show=True, save_path="trajectory_configs/trajectory_3d_plot.png")

    with open(file_path, "w") as f:
        f.write(config_str)

    print("✅ Config with multiple sections written to", file_path)

    print("✅ Config with multiple sections written to", file_name)