from dataclasses import dataclass
from typing import List, Tuple

import yaml


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
    label: str = ''  # inject section label

    def to_dict(self):
        return {
            'time': round(self.t, 2),
            'position': [round(self.x, 3), round(self.y, 3), round(self.z, 3)]
        }


@dataclass
class StringObstacle:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    label: str = ''  # inject section label

    def to_dict(self):
        return {
            'start': [round(v, 3) for v in self.start],
            'end': [round(v, 3) for v in self.end]
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

    def generate_line_strings(self, xs, ys, zs, axis='z', length=1) -> List[StringObstacle]:
        strings = []
        for x in xs:
            for y in ys:
                for z in zs:
                    if axis == 'z':
                        strings.append(StringObstacle((x, y, z - length / 2), (x, y, z + length / 2)))
                    elif axis == 'y':
                        strings.append(StringObstacle((x, y - length / 2, z), (x, y + length / 2, z)))
                    elif axis == 'x':
                        strings.append(StringObstacle((x - length / 2, y, z), (x + length / 2, y, z)))
        return strings

    def concatenate(self, other: 'TrajectoryBuilder', time_offset: float = 0.0, label: str = ''):
        for wp in other.waypoints:
            self.waypoints.append(Waypoint(wp.x, wp.y, wp.z, wp.t + time_offset, label=label))
        for s in other.strings:
            self.strings.append(StringObstacle(s.start, s.end, label=label))

    def to_yaml_config(self) -> str:
        def represent_ordered_dict(dumper, data):
            return dumper.represent_dict(data.items())

        yaml.add_representer(dict, represent_ordered_dict, Dumper=InlineListDumper)

        # Manually build the config dict with placeholders for strings/waypoints
        base_config = {
            'task_config': {
                'info_in_reset': True,
                'ctrl_freq': 60,
                'pyb_freq': 60,
                'physics': 'dyn_si_3d_delay',
                'quad_type': 9,
                'init_state': {
                    # 'init_x': 1.5,
                    # 'init_x_dot': 0,
                    # 'init_y': 1.5,
                    # 'init_y_dot': 0,
                    'init_x': 2.5,
                    'init_x_dot': 0,
                    'init_y': 1.5,
                    'init_y_dot': 0,
                    'init_z': 1.00,
                    'init_z_dot': 0,
                },
                'task': 'traj_tracking',
                'task_info': {
                    'trajectory_type': 'snap_custom',
                    'num_cycles': 1,
                    'trajectory_plane': 'xy',
                    'trajectory_position_offset': [0.0, 0.0, 0.0],
                    'trajectory_scale': 1.0,
                    'strings': '__STRINGS__',
                    'waypoints': '__WAYPOINTS__',
                },
                'inertial_prop': {'M': 0.033},
                'episode_len_sec': float(self.waypoints[-1].t) if self.waypoints else 20,
                'cost': 'quadratic',
                'constraints': [
                    {'constraint_form': 'default_constraint', 'constrained_variable': 'state'},
                    {'constraint_form': 'default_constraint', 'constrained_variable': 'input'}
                ],
                'done_on_violation': False
            }
        }

        # Dump the config excluding waypoints/strings
        config_text = yaml.dump(base_config, sort_keys=False, Dumper=InlineListDumper)

        # Group by label
        def group_with_labels(items):
            grouped = {}
            for item in items:
                grouped.setdefault(item.label or 'Unlabeled', []).append(item)
            return grouped

        def format_list_with_comments(grouped_dict, key_name):
            lines = [f'    {key_name}:']
            for label, items in grouped_dict.items():
                lines.append(f'      # === {label} ===')
                for item in items:
                    item_yaml = yaml.dump(item.to_dict(), Dumper=InlineListDumper).strip().splitlines()
                    lines.append(f'      - {item_yaml[0]}')
                    for line in item_yaml[1:]:
                        lines.append(f'        {line}')
            return lines

        string_lines = format_list_with_comments(group_with_labels(self.strings), 'strings')
        waypoint_lines = format_list_with_comments(group_with_labels(self.waypoints), 'waypoints')

        # Replace placeholders
        config_lines = config_text.splitlines()
        final_lines = []
        for line in config_lines:
            if line.strip() == 'strings: __STRINGS__':
                final_lines.extend(string_lines)
            elif line.strip() == 'waypoints: __WAYPOINTS__':
                final_lines.extend(waypoint_lines)
            else:
                final_lines.append(line)

        return '\n'.join(final_lines)


if __name__ == '__main__':
    # set_time_step = 4.0
    # set_time_step = 2.0
    # set_time_step = 1.0
    set_time_step = 0.75
    # fixed_z = 1.0
    fixed_z = 0.8
    # Section 1: Randomized obstacles + waypoints
    section1 = TrajectoryBuilder()
    # Add waypoints & obstacles for section 1
    section1.add_waypoints([
        Waypoint(x=2.5, y=1.5, z=fixed_z, t=0.0),
        Waypoint(x=2.5, y=0.0, z=fixed_z * 3 / 4, t=set_time_step),
        Waypoint(x=2.5, y=-1.5, z=fixed_z, t=set_time_step * 2),
    ])
    section1.add_strings([
        StringObstacle((3.6259018659591673, 0.0008437908662017434, 0.8210440158843995),
                       (2.398165464401245, -0.0041864584665745495, 0.8078780114650727))
        # StringObstacle((2.5, 1.0, 0.0), (2.5, 1.0, 2.0)),
        # StringObstacle((2.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
        # StringObstacle((2.0, -1.0, 0.0), (3.0, -1.0, 2.0)),
    ])
    section1.order_waypoints_by_time()

    # Section 2: Fixed string obstacles + waypoints
    section2 = TrajectoryBuilder()
    section2.add_waypoints(section2.generate_grid_waypoints(
        xs=[1.0],
        ys=[-1.5, -0.75, 0.0, 0.75, 1.5],
        zs=[fixed_z],
        time_step=set_time_step * 3 / 4))
    section2.add_waypoints(section2.generate_grid_waypoints(
        xs=[0.75],
        ys=[-1.125, 0.375],
        zs=[fixed_z],
        start_time=set_time_step * 3 / 8,
        time_step=set_time_step * 3 / 2))
    section2.add_waypoints(section2.generate_grid_waypoints(
        xs=[1.25],
        ys=[-0.375, 1.125],
        zs=[fixed_z],
        start_time=set_time_step * 9 / 8,
        time_step=set_time_step * 3 / 2))
    section2.add_strings(section2.generate_line_strings(
        xs=[1.0],
        ys=[-1.125, -0.375, 0.375, 1.125],
        zs=[fixed_z],
        axis='z'))
    section2.order_waypoints_by_time()

    # Section 3: Disturbance test obstacles + waypoints
    section3 = TrajectoryBuilder()
    section3.add_waypoints([Waypoint(x=0.5, y=1.5, z=fixed_z, t=0.0)])
    section3.add_waypoints(section3.generate_time_fixed_waypoints(
        wps=[
            (0.0, -1.5, 1.0),
            (-1.0, -0.75, 1.0),
            (-1.25, -0.00, 1.0),
            (-1.0, 0.75, 1.0),
            (0, 1.5, 1.0),
            (1.25, 1.5, 1.0),
            (2.5, 1.5, 1.0)],
        fwps=fixed_z,
        fix_axis='z',
        time_step=set_time_step / 2,
        start_time=set_time_step))
    section3.add_strings([
        StringObstacle((-0.25, -1.25, 0.0), (-0.25, -1.25, 2.0))
    ])
    final_trajectory = TrajectoryBuilder()
    section3.order_waypoints_by_time()
    # Section 1
    final_trajectory.concatenate(section1, time_offset=0, label='Section 1: Randomized String')
    offset_1 = section1.waypoints[-1].t + 1

    # Section 2
    final_trajectory.concatenate(section2, time_offset=offset_1, label='Section 2: Fixed Strings')
    offset_2 = offset_1 + section2.waypoints[-1].t + 1  # accumulate correctly

    # Section 3
    final_trajectory.concatenate(section3, time_offset=offset_2, label='Section 3: Disturbance Obstacles')
    final_trajectory.order_waypoints_by_time()
    # Export to YAML config
    config_str = final_trajectory.to_yaml_config()

    file_name = f'quadrotor_3D_attitude_tracking_{int(set_time_step)}.yaml'
    # file_name = f'quadrotor_3D_attitude_tracking_offset_{int(set_time_step)}.yaml'

    with open(file_name, 'w') as f:
        f.write(config_str)

    print('✅ Config with multiple sections written to', file_name)
