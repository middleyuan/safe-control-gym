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
        # Convert all to native Python float
        return {
            "time": float(round(self.t, 2)),
            "position": [float(round(self.x, 3)), float(round(self.y, 3)), float(round(self.z, 3))]
        }

@dataclass
class StringObstacle:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    label: str = ""  # inject section label

    def to_dict(self):
        # Convert all to native Python float
        return {
            "start": [float(round(v, 3)) for v in self.start],
            "end": [float(round(v, 3)) for v in self.end]
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
    
    def add_waypoints_scaled_time(self, wps, total_time, start_time=0.0, label=""):
        """
        Add waypoints to builder with time intervals proportional to distance, summing to total_time.
        wps: list of (x, y, z) or (x, y, z, scale) where scale is an optional multiplier for the segment time.
        """
        if len(wps) < 2:
            for wp in wps:
                x, y, z = wp[:3]
                self.waypoints.append(Waypoint(x, y, z, start_time, label=label))
            return

        # Compute distances and optional per-segment scaling
        distances = []
        scales = []
        for i in range(len(wps) - 1):
            p0 = np.array(wps[i][:3])
            p1 = np.array(wps[i+1][:3])
            scale = wps[i][3] if len(wps[i]) > 3 else 1.0
            distances.append(np.linalg.norm(p1 - p0))
            scales.append(scale)
        total_distance = sum(distances)
        if total_distance == 0:
            dts = [total_time / (len(wps)-1)] * (len(wps)-1)
        else:
            # First, compute base time allocation
            base_dts = [total_time * (d / total_distance) for d in distances]
            # Then, apply per-segment scaling
            dts = [dt * scale for dt, scale in zip(base_dts, scales)]
            # Renormalize to ensure total time matches (optional, but recommended)
            total_scaled_time = sum(dts)
            if total_scaled_time > 0:
                dts = [dt * (total_time / total_scaled_time) for dt in dts]
        t = start_time
        for i, wp in enumerate(wps):
            x, y, z = wp[:3]
            self.waypoints.append(Waypoint(x, y, z, t, label=label))
            if i < len(dts):
                t += dts[i]

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
    def add_vertical_strings(self, xs: List[float], ys: List[float], z_min: float = 0.0, z_max: float = 1.5, label: str = ""):
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
                    "init_x": 1.3,
                    "init_x_dot": 0,
                    "init_y": 0.5,
                    "init_y_dot": 0,
                    # "init_x": 2.75,
                    # "init_x_dot": 0,
                    # "init_y": 1.5,
                    # "init_y_dot": 0,
                    "init_z": 1.1,
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
    def apply_shrinking_factor(self, shrink_factor: float, preserve_z: bool = True, shrink_factor_z: float = 1.0):
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
            else:
                wp.z *= shrink_factor_z
        
        # Shrink strings
        for string in self.strings:
            # Shrink start point
            start_list = list(string.start)
            start_list[0] *= shrink_factor  # x
            start_list[1] *= shrink_factor  # y
            if not preserve_z:
                start_list[2] *= shrink_factor  # z
            else:
                start_list[2] *= shrink_factor_z
            string.start = tuple(start_list)
            
            # Shrink end point
            end_list = list(string.end)
            end_list[0] *= shrink_factor  # x
            end_list[1] *= shrink_factor  # y
            if not preserve_z:
                end_list[2] *= shrink_factor  # z
            else:
                end_list[2] *= shrink_factor_z
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

    def transform_trajectory(self, shrink_factor: float = 1.0, translation: Tuple[float, float, float] = (0.0, 0.0, 0.0), preserve_z_shrinking: bool = True, shrink_factor_z: float = 1.0):
        """
        Apply shrinking and translation in one step.
        
        Args:
            shrink_factor: Factor to shrink towards (0,0) - applied first
            translation: (dx, dy, dz) offset - applied second
            preserve_z_shrinking: If True, only shrink x,y coordinates
        """
        if shrink_factor != 1.0:
            self.apply_shrinking_factor(shrink_factor, preserve_z_shrinking, shrink_factor_z)

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
    # set_time_step = 1.0
    # set_time_step = 0.8
    set_time_step = 0.6
    # fixed_z = 1.0
    fixed_z = 0.8

    # vs11 = [1.0, 0.5]
    # vs12 = [1.0, -0.5]
    # vs21 = [0.0, 0.5]
    # vs22 = [0.0, -0.5]
    # vs31 = [-1.0, 0.5]
    # vs32 = [-1.0, -0.5]

    vs_original = {
        'vs11': [1.040841, 0.454537, 1.538180],
        'vs12': [1.040190, 0.453161, 1.537588],
        'vs21': [1.018375, -0.431456, 1.604106],
        'vs22': [1.012686, -0.439619, 0.629949],
        'vs31': [-0.000318, 0.482983, 1.548338],
        'vs32': [0.003129, 0.497312, 0.608895],
        'vs41': [0.010323, -0.508408, 1.530859],
        'vs42': [0.008381, -0.503233, 0.583155],
        'vs51': [-0.947238, 0.491893, 1.517015],
        'vs52': [-0.954391, 0.498141, 1.131038],
        'vs61': [-0.980758, -0.493244, 1.532733],
        'vs62': [-0.989345, -0.510164, 0.679069]
    }

    # Subtract 0.05 from z-coordinate (third value) of each vs
    z_adjustment = -0.1
    vs11 = [vs_original['vs11'][0], vs_original['vs11'][1], vs_original['vs11'][2] + z_adjustment]
    vs12 = [vs_original['vs12'][0], vs_original['vs12'][1], vs_original['vs12'][2] + z_adjustment]
    vs21 = [vs_original['vs21'][0], vs_original['vs21'][1], vs_original['vs21'][2] + z_adjustment]
    vs22 = [vs_original['vs22'][0], vs_original['vs22'][1], vs_original['vs22'][2] + z_adjustment]
    vs31 = [vs_original['vs31'][0], vs_original['vs31'][1], vs_original['vs31'][2] + z_adjustment]
    vs32 = [vs_original['vs32'][0], vs_original['vs32'][1], vs_original['vs32'][2] + z_adjustment]
    vs41 = [vs_original['vs41'][0], vs_original['vs41'][1], vs_original['vs41'][2] + z_adjustment]
    vs42 = [vs_original['vs42'][0], vs_original['vs42'][1], vs_original['vs42'][2] + z_adjustment]
    vs51 = [vs_original['vs51'][0], vs_original['vs51'][1], vs_original['vs51'][2] + z_adjustment]
    vs52 = [vs_original['vs52'][0], vs_original['vs52'][1], vs_original['vs52'][2] + z_adjustment]
    vs61 = [vs_original['vs61'][0], vs_original['vs61'][1], vs_original['vs61'][2] + z_adjustment]
    vs62 = [vs_original['vs62'][0], vs_original['vs62'][1], vs_original['vs62'][2] + z_adjustment]

    # Transformation parameters
    shrink_factor = 1.0  # Shrink to 60% of original size
    shrink_factor_z = 1.0  # No shrinking in z
    translation = (0.0, 0.0, 0.0)  # Translate by (0.0, 0.0, 0.0)
    preserve_z_shrinking = True  # Only shrink x,y, preserve z heights

    # Section 1: Randomized obstacles + waypoints
    section1 = TrajectoryBuilder()
    section1_wps = [
        (1.4, 0.5, 1.1), #1
        (1.4, 0.5, 1.1, 1.5), #2
        (1.4, 1.1, 1.1125, 2.0), #3
        (0.7, 1.1, 1.1125, 2.0), #4
        (0.1, 1.1, 1.125, 1.5), #5
        (0.1, 0.5, 1.125), #6
        (0.1, 0.0, 1.1),    #7
        (1.0, 0.0, 1.1, 1.2),   #8
        (1.7, -0.1, 1.05, 1.3),  #9
        (1.5, -0.9, 1.00, ),  #10
        (0.1, -0.9, 1.00),  #11
        (0.2, -0.5, 0.95, 1.3), #12
        (0.9, 0, 0.95, 1.8), #13
        (0.9, 0.5, 0.95, 1.8), #14
        (1.0, 0.8, 1.0, 1.3), #15
        (0.5, 0.8, 1.0, 1.3), #16
        (0.1, 0.8, 1.0), #17
        (0.1, 0.5, 1.0), #18
        (0.0, 0.0, 1.0, 1.3), #19
        (1.0, 0.0, 1.0), #20
        (0.9, -0.5, 1.0), #21
    ]
    # section1_total_time = 7.5  # seconds
    section1_total_time = 6.0  # seconds
    # section1_total_time = 5.0  # seconds
    section1.add_waypoints_scaled_time(section1_wps, section1_total_time, start_time=0.0, label="Section: Randomized String Obstacles")

    # section1.add_strings([
    #     StringObstacle((vs11[0], vs11[1], 1.5), 
    #                    (vs21[0], vs21[1], 0.5)),
    #     StringObstacle((vs12[0], vs12[1], 0.5), 
    #                    (vs22[0], vs22[1], 1.5)),
    #     StringObstacle((vs12[0], vs12[1], 1.5), 
    #                    (vs11[0], vs11[1], 1.5)),
    # ])

    section1.add_strings([
        StringObstacle(tuple(vs11), 
                       tuple(vs32)),
        StringObstacle((tuple(vs22)), 
                       (tuple(vs41))),
        StringObstacle(tuple(vs21), 
                       (tuple(vs11))),
    ])
    section1.add_vertical_strings(
        [vs11[0], vs21[0], vs31[0], vs41[0]],
        [vs11[1], vs21[1], vs31[1], vs41[1]]
    )
    section1.order_waypoints_by_time()

    # Section 2: Randomized obstacles + waypoints
    section2 = TrajectoryBuilder()
    section2_wps = [
        (0.9, -1.1, 1.0), #22
        (-0.1, -1.1, 1.0), #23
        (-0.4, -0.5, 1.025, 1.2), #24
        (-0.9, -0.05, 1.05), #25
        (-1.4, -0.05, 1.075), #26
        (-1.6, -0.5, 1.1), #27
        (-1.4, -1.1, 1.1), #28
        (-1.0, -1.0, 1.075), #29
        (-0.6, -0.9, 1.05), #30
        (-0.2, -0.5, 1.0), #31
        (-0.1, -0.4, 0.95), #32
        (0.2, 0.0, 0.8, 1.2), #33
        (-0.3, 0.35, 0.75, 1.3), #34
        (-0.9, 0.4, 0.7, 1.5), #35
        (-1.2, 0.35, 0.75, 1.5), #36
        (-1.5, 0.05, 0.85, 1.3), #37
        (-1.5, -0.2, 1.0, 1.2), #38
        (-1.0, -0.4, 1.1, 1.2), #39
        (-0.25, -0.25, 1.2), #40
        (0.45, 0.1, 1.15), #41
        (0.5, 0.5, 1.1), #42
        (0.5, 1.0, 1.0), #43
        (0.5, 1.0, 1.0)  #44
    ]
    # section2_total_time = 7.0  # seconds
    section2_total_time = 5.5  # seconds
    # section2_total_time = 4.5  # seconds
    section2.add_waypoints_scaled_time(section2_wps, section2_total_time, start_time=0.0, label="Section 2: Fixed Strings")

    # section2.add_strings([
    #     StringObstacle((vs21[0], vs21[1], 1.5), 
    #                    (vs32[0], vs32[1], 1.5)),
    #     StringObstacle((vs22[0], vs22[1], 0.5), 
    #                    (vs31[0], vs31[1], 1.0)),
    #     StringObstacle((vs31[0], vs31[1], 1.5), 
    #                    (vs32[0], vs32[1], 0.5)),
    # ])

    section2.add_strings([
        StringObstacle((tuple(vs31)), 
                       (tuple(vs61))),
        StringObstacle((tuple(vs42)), 
                       (tuple(vs52))),
        StringObstacle((tuple(vs51)), 
                       (tuple(vs62))),
    ])

    section2.add_vertical_strings(
        [vs51[0], vs61[0]],
        [vs51[1], vs61[1]]
    )
    section2.order_waypoints_by_time()

    # Section 3: Speed Test
    section3 = TrajectoryBuilder()
    section3_wps = [
        (-1.1, 1.1, 1.0),
        (-1.6, 0.6, 1.0),
        (-1.6, -0.6, 1.0),
        (-1.1, -1.1, 1.0),
        (1.1, -1.1, 1.0),
        (1.6, -0.6, 1.0),
        (1.6, 0.6, 1.0),
    ]
    # section3_total_time = 4.0  # seconds
    section3_total_time = 3.0  # seconds
    # section3_total_time = 2.5  # seconds
    section3.add_waypoints_scaled_time(section3_wps, section3_total_time, start_time=0.0, label="Section 3: Disturbance Obstacles")

    # Build final trajectory
    final_trajectory = TrajectoryBuilder()
    # Section 1
    final_trajectory.concatenate(section1, time_offset=0, label="Section: Randomized String Obstacles")
    offset_1 = section1.waypoints[-1].t + 0.5  # No extra time step needed

    # Section 2
    final_trajectory.concatenate(section2, time_offset=offset_1, label="Section 2: Fixed Strings")
    offset_2 = offset_1 + section2.waypoints[-1].t - section2.waypoints[0].t + 0.5 # accumulate correctly

    # Section 3
    final_trajectory.concatenate(section3, time_offset=offset_2, label="Section 3: Disturbance Obstacles")
    final_trajectory.order_waypoints_by_time()

    # Apply transformations
    final_trajectory.transform_trajectory(
        shrink_factor=shrink_factor,
        translation=translation,
        preserve_z_shrinking=preserve_z_shrinking,
        shrink_factor_z=shrink_factor_z
    )

    # Export to YAML config
    config_str = final_trajectory.to_yaml_config()

    file_name = f"final_trajectory_prototype_{float(section1_total_time)}_{float(section2_total_time)}_{float(section3_total_time)}.yaml"
    
    # Ensure the output directory exists
    output_dir = "trajectory_configs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    plot_trajectory_3d(final_trajectory, show=True, save_path="trajectory_configs/trajectory_3d_plot.png")

    with open(file_path, "w") as f:
        f.write(config_str)

    print("✅ Config with multiple sections written to", file_path)
    print("✅ Config with multiple sections written to", file_name)