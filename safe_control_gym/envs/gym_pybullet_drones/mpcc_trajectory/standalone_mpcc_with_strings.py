"""Standalone MPCC using the linear THREE_D_ATTITUDE_DELAY model (with string soft-cost).

This file is a self-contained copy of the MPCC builder adapted to use the
linear model text you pointed out (THREE_D_ATTITUDE_DELAY). It keeps the
virtual "theta" state for reference lookup (so the same path-following LUT
approach works) and adds state bounds to cap force_motor between 0.08 and 0.45 N.

Run:
    python3 tools/standalone_mpcc_with_strings.py tools/mpcc_config_with_strings.yaml
"""
from __future__ import annotations
import os
import sys
import yaml
import numpy as np
import casadi as cs
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver  # type: ignore
except Exception:
    AcadosModel = AcadosOcp = AcadosOcpSolver = None

# Use package trajectory generator for reference creation
from safe_control_gym.envs.gym_pybullet_drones.mpcc_trajectory.trajectory_generator import (
    cubic_spline_trajectory,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Default numeric parameters (from your snippet)
DEFAULT_PARAMS_ACC = [0.0905, 0.8, 0.0814]
DEFAULT_PARAMS_ROLL = [-238.1, -21.35, 179.65]
DEFAULT_PARAMS_PITCH = [-238.1, -21.35, 179.65]
DEFAULT_PARAMS_YAW = [-170.4, -22.22, 280]
# physical thrust caps (as in quadrotor._set_action_space)
THRUST_MIN = 0.08
THRUST_MAX = 0.45


def export_quadrotor_ode_model(prior_prop: dict | None = None) -> "AcadosModel | None":
    """Create AcadosModel for the linear THREE_D_ATTITUDE_DELAY-like model.

    States (14):
      [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot, force_motor, theta_phase]
    Inputs (5):
      [T_c, R_c, P_c, Y_c, v_theta]   (v_theta is theta_phase_dot)
    """
    if prior_prop is None:
        prior_prop = {}

    params_acc = prior_prop.get("param_acc", DEFAULT_PARAMS_ACC)
    params_roll_rate = prior_prop.get("params_roll_rate", DEFAULT_PARAMS_ROLL)
    params_pitch_rate = prior_prop.get("params_pitch_rate", DEFAULT_PARAMS_PITCH)
    params_yaw_rate = prior_prop.get("params_yaw_rate", DEFAULT_PARAMS_YAW)

    model_name = "force_mpcc_linear_delay"

    # symbolic states
    x = cs.MX.sym("x")
    x_dot = cs.MX.sym("x_dot")
    y = cs.MX.sym("y")
    y_dot = cs.MX.sym("y_dot")
    z = cs.MX.sym("z")
    z_dot = cs.MX.sym("z_dot")
    phi = cs.MX.sym("phi")
    theta = cs.MX.sym("theta")
    psi = cs.MX.sym("psi")
    phi_dot = cs.MX.sym("phi_dot")
    theta_dot = cs.MX.sym("theta_dot")
    psi_dot = cs.MX.sym("psi_dot")
    force_motor = cs.MX.sym("force_motor")  # last physical motor force state
    theta_phase = cs.MX.sym("theta_phase")  # virtual phase for LUT reference

    states = cs.vertcat(
        x,
        x_dot,
        y,
        y_dot,
        z,
        z_dot,
        phi,
        theta,
        psi,
        phi_dot,
        theta_dot,
        psi_dot,
        force_motor,
        theta_phase,
    )

    # inputs: T_c, R_c, P_c, Y_c, v_theta
    T_c = cs.MX.sym("T_c")
    R_c = cs.MX.sym("R_c")
    P_c = cs.MX.sym("P_c")
    Y_c = cs.MX.sym("Y_c")
    v_theta = cs.MX.sym("v_theta")
    inputs = cs.vertcat(T_c, R_c, P_c, Y_c, v_theta)

    # Forces disturbance placeholder (not used here)
    fx = cs.MX.sym("fx")
    fy = cs.MX.sym("fy")
    fz = cs.MX.sym("fz")
    forces_dist = cs.vertcat(fx, fy, fz)

    # simple gravity + mass constants placeholder; user can override via prior_prop if needed
    MASS = prior_prop.get("M", 0.037)
    GRAVITY_VEC = cs.vertcat(0, 0, -9.81)

    # Linear model dynamics from the snippet (mirrored exactly):
    # normalization mapping parameters for df calculation (kept as in snippet)
    cmd_min = -1
    cmd_max = 1
    f_min = -1
    f_max = 1

    dT_c = 2 * (T_c - cmd_min) / (cmd_max - cmd_min) - 1
    df = 2 * (force_motor - f_min) / (f_max - f_min) - 1
    df_dot = (params_acc[1] * (dT_c + params_acc[0]) - df) / params_acc[2]

    # Compute acceleration contributions (using full 3D rotation expressions)
    x_ddot = (1.0 / MASS) * force_motor * (
        cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)
    )
    y_ddot = (1.0 / MASS) * force_motor * (
        cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)
    )
    z_ddot = (1.0 / MASS) * force_motor * cs.cos(phi) * cs.cos(theta) + GRAVITY_VEC[2]  # GRAVITY_VEC[2] is -g

    # rotational rate dynamics (linear PD-like mapping from snippet)
    phi_ddot = params_roll_rate[0] * phi + params_roll_rate[1] * phi_dot + params_roll_rate[2] * R_c
    theta_ddot = params_pitch_rate[0] * theta + params_pitch_rate[1] * theta_dot + params_pitch_rate[2] * P_c
    psi_ddot = params_yaw_rate[0] * psi + params_yaw_rate[1] * psi_dot + params_yaw_rate[2] * Y_c

    # force_motor derivative from snippet scaled back to original variable
    # snippet set X_dot last element to (f_max - f_min)/2 * df_dot
    force_motor_dot = (f_max - f_min) / 2.0 * df_dot

    # phase derivative
    theta_phase_dot = v_theta

    states_dot = cs.vertcat(
        x_dot,
        x_ddot,
        y_dot,
        y_ddot,
        z_dot,
        z_ddot,
        phi_dot,
        theta_dot,
        psi_dot,
        phi_ddot,
        theta_ddot,
        psi_ddot,
        force_motor_dot,
        theta_phase_dot,
    )

    if AcadosModel is None:
        # acados not available on this machine; return CASADI-based model placeholder
        return None

    model = AcadosModel()
    model.name = model_name
    model.x = states
    model.u = inputs
    model.p = forces_dist
    model.f_expl_expr = states_dot
    model.f_impl_expr = None
    return model


def create_ocp_solver(
    Tf: float,
    N: int,
    waypoints: dict[str, NDArray],
    strings: list[dict] | None = None,
    prior_prop: dict | None = None,
    verbose: bool = False,
    weights: dict | None = None,
    x0: np.ndarray | None = None,  
) -> tuple["AcadosOcpSolver | None", "AcadosOcp | None"]:
    """Create Acados OCP solver using the linear THREE_D_ATTITUDE_DELAY model.

    Adds per-string soft penalty (same as prior standalone). Also enforces
    bounds on the force_motor state (THRUST_MIN..THRUST_MAX).
    """
    if AcadosOcp is None:
        print("[WARN] acados_template not available: OCP construction skipped.")
        return None, None

    ocp = AcadosOcp()
    model = export_quadrotor_ode_model(prior_prop=prior_prop)
    if model is None:
        print("[WARN] Could not build AcadosModel (acados_template missing).")
        return None, None
    ocp.model = model

    x = model.x
    u = model.u
    MASS = prior_prop.get("M", 0.037)
    hover_force = MASS * 9.81            # ≈ 0.363 N
    force_ref = hover_force  
    # pos as first three states
    pos = cs.vertcat(x[0], x[2], x[4])
    # theta_phase is last state (index 13)
    theta_phase = x[13]
    
    
    ocp.solver_options.N_horizon = N

    # Reference LUTs (linear interpolants over theta_phase)
    lut_px = cs.interpolant("lut_px", "linear", [waypoints["time"]], waypoints["pos"][:, 0])
    lut_py = cs.interpolant("lut_py", "linear", [waypoints["time"]], waypoints["pos"][:, 1])
    lut_pz = cs.interpolant("lut_pz", "linear", [waypoints["time"]], waypoints["pos"][:, 2])
    pos_ref = cs.vertcat(lut_px(theta_phase), lut_py(theta_phase), lut_pz(theta_phase))
    params_acc_local = prior_prop.get("param_acc", DEFAULT_PARAMS_ACC)
    u_ref = cs.DM([hover_force, 0.0, 0.0, 0.0, 1.0])  # [T_c, R_c, P_c, Y_c, v_theta]
    # approximate velocity reference via gradient of interpolant wrt phase
    vel_ref = cs.vertcat(
        cs.gradient(pos_ref[0], theta_phase),
        cs.gradient(pos_ref[1], theta_phase),
        cs.gradient(pos_ref[2], theta_phase),
    )

    # simple nominal input reference
    if weights is None:
        weights = {
            'contouring': 50.0,
            'lag': 100.0,
            'input': 0.01,
            'progress': 5.0,
            'string': 50.0,
            'string_radius': 0.1,
            'force': 50.0,   # stronger keep near hover
        }

    # Define cost weights
    Q_force = weights['force']
    Q_c = np.eye(3) * weights['contouring']  # contouring error (waypoint tracking)
    Q_l = np.eye(3) * weights['lag']         # lag error
    mu = weights['progress']                  # progress weight
    R = np.diag(
        [
            3.0,  # rpy
            3.0,  # rpy
            3.0,  # rpy
            1.0,  # thrust
            1.0,  # v_theta
        ]
    )   # input cost (all inputs weighted equally)

    # contouring/lag errors (same construction)
    e = cs.vertcat(pos - pos_ref)
    normal = vel_ref
    normal = normal / cs.norm_2(normal + 1e-9)
    e_l_hat = cs.dot(normal, e) * normal
    e_c_hat = e - e_l_hat

    v_theta = u[4]  # v_theta is input index 4

    # base stage cost
    stage_cost = (
        e_c_hat.T @ Q_c @ e_c_hat
        + e_l_hat.T @ Q_l @ e_l_hat
        - mu * v_theta
        + (u - u_ref).T @ R @ (u - u_ref)
    )
    force_motor_state = x[12]
    # stage_cost = stage_cost + Q_force * (force_motor_state - force_ref)**2
    # Add string avoidance soft cost
    if strings:
        for s in strings:
            a = cs.DM(s["start"])
            b = cs.DM(s["end"])
            ab = b - a
            ab_norm_sq = cs.dot(ab, ab) + 1e-8
            t_proj = cs.dot((pos - a), ab) / ab_norm_sq
            t_clamped = cs.fmin(cs.fmax(t_proj, 0.0), 1.0)
            closest = a + t_clamped * ab
            dist = cs.norm_2(pos - closest)
            r_safe = float(s.get("radius", weights['string_radius']))
            w = float(s.get("weight", weights['string']))
            penalty = (cs.fmax(r_safe - dist, 0.0)) ** 2 * w
            stage_cost = stage_cost + penalty

    terminal_cost = e_c_hat.T @ Q_c @ e_c_hat + e_l_hat.T @ Q_l @ e_l_hat
    ocp.model.cost_expr_ext_cost = stage_cost
    ocp.model.cost_expr_ext_cost_e = terminal_cost

    # Input bounds (assume controller commands T_c in [-1,1], angle commands reasonable)
    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0, 0.0, -4.0])  # keep thrust_cmd lower bound 0
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, 4.0, 4.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # Force_motor state bounds (enforce physical thrust limits)
    # acados expects idxbx, lbx, ubx arrays
    ocp.constraints.idxbx = np.array([12], dtype=int)  # index of force_motor in state vector
    ocp.constraints.lbx = np.array([THRUST_MIN])
    ocp.constraints.ubx = np.array([THRUST_MAX])

    if x0 is None:
        # Fallback (but warn) – better to always supply x0
        print("[WARN] x0 not provided; using zeros (will force initial force=0).")
        x0 = np.zeros((x.rows(),))
    ocp.constraints.x0 = x0  # equality constraint at stage 0
    print(f"[DEBUG] OCP initial constraint x0[0:5]: {ocp.constraints.x0[:5]}")
    print(f"[DEBUG] OCP initial force_motor x0[12]: {ocp.constraints.x0[12]:.3f}")

    ocp.parameter_values = np.zeros(model.p.rows())

    # # solver options (copied)
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    # ocp.solver_options.hpipm_mode = "ROBUST"
    # ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # ocp.solver_options.tol = 1e-5
    # ocp.solver_options.tf = Tf
    # ocp.solver_options.warm_start = True
    # ocp.solver_options.qp_solver_iter_max = 20
    # ocp.solver_options.nlp_solver_max_iter = 50
    # ocp.solver_options.qp_solver_cond_N = N
    
    # Modified solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hpipm_mode = "ROBUST"  # More robust but slower
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # Full SQP instead of RTI
    ocp.solver_options.nlp_solver_max_iter = 100  # More iterations
    ocp.solver_options.tol = 1e-4  # Slightly relaxed tolerance
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_iter_max = 50  # More QP iterations
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.print_level = 1

    try:
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=f"c_generated_code/{model.name}.json", verbose=verbose)
        print("[DEBUG] Solver built. Equality x0 force =", ocp.constraints.x0[12])
    except Exception as e:
        print(f"[WARN] Could not instantiate AcadosOcpSolver: {e}")
        acados_ocp_solver = None

    return acados_ocp_solver, ocp


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_task_config(cfg: dict):
    """Parse either top-level simple config or task_config format."""
    # defaults
    strings = None
    prior_prop = None
    frequency = 200.0
    desired_completion_time = None

    if "task_config" in cfg:
        tc = cfg["task_config"]
        ti = tc.get("task_info", {})
        wp_entries = ti.get("waypoints", [])
        if not wp_entries:
            # try alternative field names (some configs store raw list under 'waypoints' top-level)
            raise ValueError("No waypoints found under task_config.task_info.waypoints")
        # entries might be dicts with 'position' and optional 'time', or plain lists
        positions = []
        times = []
        for e in wp_entries:
            if isinstance(e, dict) and "position" in e:
                positions.append(np.array(e["position"], dtype=float))
                times.append(float(e.get("time", 0.0)))
            elif isinstance(e, (list, tuple, np.ndarray)) and len(e) >= 3:
                positions.append(np.array(e[:3], dtype=float))
                times.append(float(0.0))
            else:
                raise ValueError("Unsupported waypoint entry format in task_info.waypoints")
        positions = np.vstack(positions)
        # if all times are zero, let desired_completion_time control sampling
        desired_completion_time = float(tc.get("episode_len_sec", max(times) if max(times) > 0 else len(positions)))
        frequency = float(tc.get("ctrl_freq", tc.get("pyb_freq", frequency)))
        strings = ti.get("strings", None)
        prior_prop = tc.get("inertial_prop", None)
    else:
        # legacy simple format
        if "waypoints" in cfg:
            positions = np.array(cfg["waypoints"], dtype=float)
            desired_completion_time = float(cfg.get("desired_completion_time", positions.shape[0]))
            frequency = float(cfg.get("frequency", 200.0))
            strings = cfg.get("strings", None)
            prior_prop = cfg.get("prior_prop", None)
        else:
            raise ValueError("Config file does not contain recognized waypoint definitions")

    return positions, desired_completion_time, frequency, strings, prior_prop

def _plot_xy_with_solution(
    positions: np.ndarray,
    way: dict,
    solution: np.ndarray | None,
    save_dir: str,
    strings: list | None = None
):
    """Plot XY view with waypoints, reference trajectory, strings, and MPCC solution."""
    plt.figure(figsize=(8, 6))
    
    # Plot waypoints and their connections
    plt.plot(positions[:, 0], positions[:, 1], "-k", alpha=0.3, label='Waypoint connections')
    plt.scatter(positions[:, 0], positions[:, 1], c="r", label='Waypoints')
    for i, wp in enumerate(positions):
        plt.text(wp[0], wp[1], str(i + 1))
    
    # Plot cubic spline reference
    plt.plot(way['pos'][:, 0], way['pos'][:, 1], 'g-', alpha=0.8, label='Reference spline')
    
    # Plot MPCC solution if available
    if solution is not None:
        plt.plot(solution[:, 0], solution[:, 1], '--b', label='MPCC solution')
    
    # Plot strings if provided
    if strings:
        for s in strings:
            a = s["start"]; b = s["end"]
            plt.plot([a[0], b[0]], [a[1], b[1]], color="blue", linewidth=2)
            # Mark perpendicular strings
            if np.isclose(a[0], b[0]) and np.isclose(a[1], b[1]):
                plt.scatter(a[0], a[1], s=120, color="blue")
    
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("XY Trajectory")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trajectory_xy.png"), dpi=300)
    plt.close()

def _plot_solutions(
    positions: np.ndarray,
    way: dict,
    solution: np.ndarray | None,
    full_state_solution: np.ndarray | None,
    input_solution: np.ndarray | None,
    save_dir: str,
    strings: list | None = None
):
    """Plot XY and XZ views plus state trajectories."""
    # Spatial plots (XY and XZ)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # XY plot
    ax1.plot(positions[:, 0], positions[:, 1], "-k", alpha=0.3, label='Waypoint connections')
    ax1.scatter(positions[:, 0], positions[:, 1], c="r", label='Waypoints')
    for i, wp in enumerate(positions):
        ax1.text(wp[0], wp[1], str(i + 1))
    ax1.plot(way['pos'][:, 0], way['pos'][:, 1], 'g-', alpha=0.8, label='Reference spline')
    if solution is not None:
        ax1.plot(solution[:, 0], solution[:, 1], '--b', label='MPCC solution')
    if strings:
        for s in strings:
            a = s["start"]; b = s["end"]
            ax1.plot([a[0], b[0]], [a[1], b[1]], color="blue", linewidth=2)
    ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]")
    ax1.set_title("XY View"); ax1.grid(True)
    ax1.legend(); ax1.axis('equal')
    
    # XZ plot
    ax2.plot(positions[:, 0], positions[:, 2], "-k", alpha=0.3, label='Waypoint connections')
    ax2.scatter(positions[:, 0], positions[:, 2], c="r", label='Waypoints')
    for i, wp in enumerate(positions):
        ax2.text(wp[0], wp[2], str(i + 1))
    ax2.plot(way['pos'][:, 0], way['pos'][:, 2], 'g-', alpha=0.8, label='Reference spline')
    if solution is not None:
        ax2.plot(solution[:, 0], solution[:, 2], '--b', label='MPCC solution')
    if strings:
        for s in strings:
            a = s["start"]; b = s["end"]
            ax2.plot([a[0], b[0]], [a[2], b[2]], color="blue", linewidth=2)
    ax2.set_xlabel("X [m]"); ax2.set_ylabel("Z [m]")
    ax2.set_title("XZ View"); ax2.grid(True)
    ax2.legend(); ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trajectories_spatial.png"), dpi=300)
    plt.close()

    # State trajectories (if full solution available)
    if full_state_solution is not None and input_solution is not None:
        t = np.linspace(0, 1.0, len(full_state_solution))
        
        # Create 3x2 subplot for states and inputs
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Position and velocity
        axes[0,0].plot(t, full_state_solution[:, 0], 'b-', label='x')
        axes[0,0].plot(t, full_state_solution[:, 2], 'g-', label='y')
        axes[0,0].plot(t, full_state_solution[:, 4], 'r-', label='z')
        axes[0,0].set_ylabel('Position [m]')
        axes[0,0].grid(True); axes[0,0].legend()
        
        axes[0,1].plot(t, full_state_solution[:, 1], 'b--', label='vx')
        axes[0,1].plot(t, full_state_solution[:, 3], 'g--', label='vy')
        axes[0,1].plot(t, full_state_solution[:, 5], 'r--', label='vz')
        axes[0,1].set_ylabel('Velocity [m/s]')
        axes[0,1].grid(True); axes[0,1].legend()
        
        # Angles and rates
        axes[1,0].plot(t, np.rad2deg(full_state_solution[:, 6]), 'b-', label='roll')
        axes[1,0].plot(t, np.rad2deg(full_state_solution[:, 7]), 'g-', label='pitch')
        axes[1,0].plot(t, np.rad2deg(full_state_solution[:, 8]), 'r-', label='yaw')
        axes[1,0].set_ylabel('Angle [deg]')
        axes[1,0].grid(True); axes[1,0].legend()
        
        axes[1,1].plot(t, np.rad2deg(full_state_solution[:, 9]), 'b--', label='roll rate')
        axes[1,1].plot(t, np.rad2deg(full_state_solution[:, 10]), 'g--', label='pitch rate')
        axes[1,1].plot(t, np.rad2deg(full_state_solution[:, 11]), 'r--', label='yaw rate')
        axes[1,1].set_ylabel('Angular Rate [deg/s]')
        axes[1,1].grid(True); axes[1,1].legend()
        
        # Force and inputs
        axes[2,0].plot(t, full_state_solution[:, 12], 'k-', label='force')
        axes[2,0].axhline(y=0.08, color='r', linestyle=':', label='min thrust')
        axes[2,0].axhline(y=0.45, color='r', linestyle=':', label='max thrust')
        axes[2,0].set_ylabel('Force [N]')
        axes[2,0].grid(True); axes[2,0].legend()
        
        axes[2,1].plot(t, input_solution[:, 0], 'k-', label='T_c')
        axes[2,1].plot(t, input_solution[:, 1], 'b-', label='R_c')
        axes[2,1].plot(t, input_solution[:, 2], 'g-', label='P_c')
        axes[2,1].plot(t, input_solution[:, 3], 'r-', label='Y_c')
        axes[2,1].plot(t, input_solution[:, 4], 'm-', label='v_theta')
        axes[2,1].set_ylabel('Control Inputs')
        axes[2,1].grid(True); axes[2,1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "kinematics.png"), dpi=300)
        plt.close()

def _plot_mpcc_only(
    solution: np.ndarray | None,
    save_dir: str,
    strings: list | None = None
):
    """Plot only the MPCC solution in XY and XZ views."""
    if solution is None:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # XY plot
    ax1.plot(solution[:, 0], solution[:, 1], 'b-', linewidth=2, label='MPCC solution')
    if strings:
        for s in strings:
            a = s["start"]; b = s["end"]
            ax1.plot([a[0], b[0]], [a[1], b[1]], color='red', linewidth=2, label='String')
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title("XY View - MPCC Solution")
    ax1.grid(True)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicate labels
    ax1.legend(by_label.values(), by_label.keys())
    ax1.axis('equal')
    
    # XZ plot
    ax2.plot(solution[:, 0], solution[:, 2], 'b-', linewidth=2, label='MPCC solution')
    if strings:
        for s in strings:
            a = s["start"]; b = s["end"]
            ax2.plot([a[0], b[0]], [a[2], b[2]], color='red', linewidth=2, label='String')
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Z [m]")
    ax2.set_title("XZ View - MPCC Solution")
    ax2.grid(True)
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mpcc_solution_only.png"), dpi=300)
    plt.close()

def main(config_path: str):
    # Load config
    cfg = load_yaml(config_path)
    positions, desired_completion_time, frequency, strings, prior_prop = _parse_task_config(cfg)

    # Generate spline trajectory
    way = cubic_spline_trajectory(positions, desired_completion_time, frequency, overhang=0, plot=False)
    print(f"[INFO] Generated spline trajectory with {len(way['time'])} samples")
    mass = (prior_prop.get("M") if prior_prop else 0.037)
    hover_thrust = mass * 9.81
    # Build initial state with correct indexing: x, x_dot, y, y_dot, z, z_dot, ...
    x0 = np.zeros(14)
    x0[0] = positions[0][0]      # x
    x0[2] = positions[0][1]      # y
    x0[4] = positions[0][2]      # z
    x0[12] = 0.364               # force_motor near hover (user request)
    x0[13] = way['time'][0]      # theta_phase
    # MPCC settings
    Tf = 4.0  # Longer horizon to see more of the path
    N = 50    # More discretization points
            
    # Adjusted weights - increase tracking importance
    weights = {
        'contouring': 50.0,
        'lag': 100.0,
        'input': 0.01,
        'progress': 5.0,
        'string': 50.0,
        'string_radius': 0.1,
        'force': 800.0,   # stronger keep near hover
    }

    # Create output directory
    save_dir = cfg.get("output_dir", "obstacle_course_warmstart_data/trajectory_20.5")
    os.makedirs(save_dir, exist_ok=True)

    # Create and run solver
    solver, ocp = create_ocp_solver(
        Tf, N, way, strings=strings, 
        prior_prop=prior_prop, 
        verbose=True,
        weights=weights,
        x0=x0
    )

    solution = None
    full_state_solution = None
    input_solution = None
    
    if solver is not None:
        print("[INFO] Running MPCC solver...")
        try:
            print(f"[DEBUG] First waypoint (x,y,z): {positions[0]}")
            hover_thrust = (prior_prop.get("M", 0.037) if prior_prop else 0.037) * 9.81
            print(f"[DEBUG] Computed hover thrust: {hover_thrust:.4f} N")

            x0 = np.zeros(14)
            # Correct index mapping: x=0, y=2, z=4
            x0[0] = positions[0][0]
            x0[2] = positions[0][1]
            x0[4] = positions[0][2]
            x0[12] = 0.36           # start force_motor near hover (user request)
            x0[13] = way['time'][0] # phase
            hover_force = 0.36
            for i in range(N+1):
                # ... existing state guess code ...
                if i < N:
                    solver.set(i, "u", np.array([hover_force, 0.0, 0.0, 0.0, 1.0]))
            print(f"[DEBUG] Steady-state T_c (hover) = {hover_force:.4f}")
            print(f"[DEBUG] Initial state set: x={x0[0]:.3f}, y={x0[2]:.3f}, z={x0[4]:.3f}, force={x0[12]:.3f}")

            solver.set(0, "x", x0)

            # Initialize guesses for all nodes
            for i in range(N+1):
                alpha = i / N
                idx = int(alpha * (len(way['time']) - 1))
                guess = np.zeros(14)
                guess[0] = way['pos'][idx, 0]
                guess[2] = way['pos'][idx, 1]
                guess[4] = way['pos'][idx, 2]
                guess[12] = 0.36
                guess[13] = way['time'][idx]
                solver.set(i, "x", guess)
                if i < N:
                    # Nominal small inputs, progress v_theta=1
                    solver.set(i, "u", np.array([hover_force, 0.0, 0.0, 0.0, 1.0]))
                if i in (0, N):
                    print(f"[DEBUG] Guess node {i}: x={guess[0]:.3f}, y={guess[2]:.3f}, z={guess[4]:.3f}")

            status = solver.solve()
            print(f"[INFO] Solver status: {status}")
            if status != 0:
                print("[WARN] Solver did not converge optimally.")
            
            full_state_solution = np.zeros((N+1, 14))
            solution = np.zeros((N+1, 3))
            input_solution = np.zeros((N+1, 5))

            for i in range(N+1):
                st = solver.get(i, "x")
                full_state_solution[i] = st
                solution[i] = [st[0], st[2], st[4]]  # x,y,z extraction
                if i < N:
                    input_solution[i] = solver.get(i, "u")
            print(f"[DEBUG] Extracted start pos: {solution[0]}")
            print(f"[DEBUG] Extracted end pos: {solution[-1]}")
            print(f"[DEBUG] z range: min={solution[:,2].min():.3f}, max={solution[:,2].max():.3f}")
            print(f"[DEBUG] force_motor range: min={full_state_solution[:,12].min():.3f}, max={full_state_solution[:,12].max():.3f}")

            np.savez_compressed(
                os.path.join(save_dir, "mpcc_solution.npz"),
                positions=solution,
                full_state=full_state_solution,
                inputs=input_solution,
                status=status
            )

            _plot_solutions(positions, way, solution, full_state_solution, input_solution, save_dir, strings=strings)
            _plot_mpcc_only(solution, save_dir, strings=strings)
            print("[INFO] Plots generated.")
        except Exception as e:
            print(f"[ERROR] Solver execution failed: {e}")
            import traceback; traceback.print_exc()
    
    if solution is not None:
        try:
            _plot_solutions(positions, way, solution, full_state_solution, input_solution, save_dir, strings=strings)
            _plot_mpcc_only(solution, save_dir, strings=strings)
            print("[INFO] Created plots successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create plots: {e}")
            import traceback
            traceback.print_exc()

    print(f"[INFO] Process complete. Output directory: {os.path.abspath(save_dir)}")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/mpcc_trajectory_generator.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])