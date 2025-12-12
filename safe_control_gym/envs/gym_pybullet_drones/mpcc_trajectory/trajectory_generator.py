"""Different Trajectory Generators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from numpy.typing import NDArray


def cubic_spline_trajectory(
    waypoints: NDArray,
    desired_completion_time: float,
    frequency: float,
    overhang: int = 0,
    plot: bool = False,
) -> dict[str, NDArray]:
    """TODO.

    Args:
        waypoints (NDArray): Nx3 position waypoints
        desired_completion_time (float): in seconds
        frequency (float): How fast the spline is sampled
        overhang (int): How often the last waypoint gets appended to the waypoints
        plot (bool): If the results should be plotted

    Returns:
        tuple[NDArray, NDArray, NDArray]: Interpolated waypoints in position, velocity, and yaw
    """
    ts = np.linspace(0, desired_completion_time, np.shape(waypoints)[0])
    # boundary conditions are positive vel in z at start and no acc at end
    cs_px = CubicSpline(ts, waypoints[:, 0], bc_type=((1, 0.0), (2, 0.0)))
    cs_py = CubicSpline(ts, waypoints[:, 1], bc_type=((1, 0.0), (2, 0.0)))
    cs_pz = CubicSpline(ts, waypoints[:, 2], bc_type=((1, 0.1), (2, 0.0)))
    cs_vx = cs_px.derivative()
    cs_vy = cs_py.derivative()
    cs_vz = cs_pz.derivative()

    ts = np.linspace(0, desired_completion_time, int(frequency * desired_completion_time))
    px_des = cs_px(ts)
    py_des = cs_py(ts)
    pz_des = cs_pz(ts)
    vx_des = cs_vx(ts)
    vy_des = cs_vy(ts)
    vz_des = cs_vz(ts)

    waypoints_pos = np.stack((px_des, py_des, pz_des)).T
    waypoints_vel = np.stack((vx_des, vy_des, vz_des)).T
    waypoints_yaw = np.zeros_like(px_des)
    # waypoints_yaw = np.arctan2(vy_des, vx_des) # Heading into velocity direction

    # Stacking overhang points, might be needed for MPC
    if overhang > 0:
        # waypoints_pos = np.concat((waypoints_pos, [waypoints_pos[-1]] * overhang))
        waypoints_vel = np.concat((waypoints_vel, [waypoints_vel[-1]] * overhang))
        waypoints_yaw = np.concat((waypoints_yaw, [waypoints_yaw[-1]] * overhang))
        for i in range(overhang):
            waypoints_pos = np.concat(
                (waypoints_pos, [waypoints_pos[-1] + waypoints_vel[-1] / frequency])
            )

    waypoints_time = np.linspace(
        0, desired_completion_time + overhang / frequency, len(waypoints_pos)
    )

    if plot:
        print(f"{waypoints=}")

        # 2D Plot over time
        fig, axs = plt.subplots(3)
        axs[0].plot(waypoints_time, waypoints_pos[:, 0], label="Pos x")
        axs[0].plot(waypoints_time, waypoints_pos[:, 1], label="Pos y")
        axs[0].plot(waypoints_time, waypoints_pos[:, 2], label="Pos z")
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Pos [m]")
        axs[1].plot(waypoints_time, waypoints_vel[:, 0], label="Vel x")
        axs[1].plot(waypoints_time, waypoints_vel[:, 1], label="Vel y")
        axs[1].plot(waypoints_time, waypoints_vel[:, 2], label="Vel z")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Vel [m/s]")
        axs[2].plot(waypoints_time, waypoints_yaw, label="yaw")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Angle [rad]")
        for ax in axs.flat:
            ax.legend()
            ax.grid()

        # 2D Plot trajectory
        fig, axs = plt.subplots(2)
        axs[0].plot(
            waypoints_pos[:-overhang, 0], waypoints_pos[:-overhang, 2], label="Trajectory xz"
        )
        axs[0].plot(waypoints_pos[-overhang:, 0], waypoints_pos[-overhang:, 2], label="Stopping")
        axs[0].set_xlabel("Pos x [m]")
        axs[0].set_ylabel("Pos z [m]")
        axs[1].plot(
            waypoints_pos[:-overhang, 0], waypoints_pos[:-overhang, 1], label="Trajectory xy"
        )
        axs[1].plot(waypoints_pos[-overhang:, 0], waypoints_pos[-overhang:, 1], label="Stopping")
        axs[1].set_xlabel("Pos x [m]")
        axs[1].set_ylabel("Pos y [m]")
        for ax in axs.flat:
            ax.legend()
            ax.grid()

        # 3D Plot trajectory
        ax = plt.figure().add_subplot(projection="3d")
        ax.plot(
            waypoints_pos[:-overhang, 0],
            waypoints_pos[:-overhang, 1],
            waypoints_pos[:-overhang, 2],
            label="Trajectory",
        )
        ax.plot(
            waypoints_pos[-overhang:, 0],
            waypoints_pos[-overhang:, 1],
            waypoints_pos[-overhang:, 2],
            label="Stopping",
        )
        lim = np.max(np.max(waypoints_pos))
        axis_length = 0.1 * lim
        ax.plot([0, axis_length], [0, 0], [0, 0], color="r")
        ax.plot([0, 0], [0, axis_length], [0, 0], color="g")
        ax.plot([0, 0], [0, 0], [0, axis_length], color="b")
        ax.legend()
        ax.set_xlim3d(-lim, lim)
        ax.set_ylim3d(-lim, lim)
        ax.set_zlim3d(0, lim)
        plt.show()

    waypoints = {
        "time": waypoints_time,
        "pos": waypoints_pos,
        "vel": waypoints_vel,
        "yaw": waypoints_yaw,
    }

    return waypoints