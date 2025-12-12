"""This module implements an example MPCC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver  # type: ignore
from lsy_models.utils.constants import Constants

from .trajectory_generator import cubic_spline_trajectory
from .controller import Controller
from . import force_predictor as pred
if TYPE_CHECKING:
    from numpy.typing import NDArray

constants = Constants.from_config("cf2x_L250")

ACADOS_STATUS = {
    0: "ACADOS_SUCCESS: Solution found",
    1: "ACADOS_MAXITER: Maximum iterations reached",
    2: "ACADOS_MINSTEP: Minimum step size reached",
    3: "ACADOS_QP_FAILURE: QP solver failed",
    4: "ACADOS_NLP_FAILURE: NLP solver failed",
    5: "ACADOS_REGULARIZATION_FAILURE: Regularization error",
}


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "force_mpcc"

    """Model setting"""
    # define basic variables in state and input vector
    px, py, pz = cs.MX.sym("px"), cs.MX.sym("py"), cs.MX.sym("pz")
    pos = cs.vertcat(px, py, pz)  # Position
    vx, vy, vz = cs.MX.sym("vx"), cs.MX.sym("vy"), cs.MX.sym("vz")
    vel = cs.vertcat(vx, vy, vz)  # Velocity
    roll, pitch, yaw = cs.MX.sym("roll"), cs.MX.sym("pitch"), cs.MX.sym("yaw")
    rpy = cs.vertcat(roll, pitch, yaw)  # Euler angles
    droll, dpitch, dyaw = cs.MX.sym("droll"), cs.MX.sym("dpitch"), cs.MX.sym("dyaw")
    rpy_dot = cs.vertcat(droll, dpitch, dyaw)  # Euler angles dot
    thrust = cs.MX.sym("thrust")
    theta = cs.MX.sym("theta")

    r_cmd = cs.MX.sym("r_cmd")
    p_cmd = cs.MX.sym("p_cmd")
    y_cmd = cs.MX.sym("y_cmd")
    thrust_cmd = cs.MX.sym("thrust_cmd")
    v_theta = cs.MX.sym("v_theta")  # Needed for contour progress

    # define state and input vector
    states = cs.vertcat(pos, rpy, vel, rpy_dot, thrust, theta)
    inputs = cs.vertcat(r_cmd, p_cmd, y_cmd, thrust_cmd, v_theta)

    # define external disturbances
    fx, fy, fz = cs.MX.sym("fx"), cs.MX.sym("fy"), cs.MX.sym("fz")
    forces_dist = cs.vertcat(fx, fy, fz)  # Disturbance forces
    # tx, ty, tz = cs.MX.sym("tx"), cs.MX.sym("ty"), cs.MX.sym("tz")
    # torques_dist = cs.vertcat(tx, ty, tz)  # Disturbance torques

    # Define nonlinear system dynamics
    pos_dot = vel
    rpy_dot = cs.vertcat(droll, dpitch, dyaw)
    x_axis = cs.vertcat(cs.cos(pitch) * cs.cos(yaw), cs.cos(pitch) * cs.sin(yaw), -cs.sin(pitch))
    y_axis = cs.vertcat(
        cs.sin(roll) * cs.sin(pitch) * cs.cos(yaw) - cs.cos(roll) * cs.sin(yaw),
        cs.sin(roll) * cs.sin(pitch) * cs.sin(yaw) + cs.cos(roll) * cs.cos(yaw),
        cs.sin(roll) * cs.cos(pitch),
    )
    z_axis = cs.vertcat(
        cs.cos(roll) * cs.sin(pitch) * cs.cos(yaw) + cs.sin(roll) * cs.sin(yaw),
        cs.cos(roll) * cs.sin(pitch) * cs.sin(yaw) - cs.sin(roll) * cs.cos(yaw),
        cs.cos(roll) * cs.cos(pitch),
    )
    R = cs.horzcat(x_axis, y_axis, z_axis)  # Rotation matrix from body to inertial frame
    thrust_scaled = constants.DI_DD_ACC[0] * thrust
    vel_dot = (
        z_axis * thrust_scaled / constants.MASS
        + constants.GRAVITY_VEC
        + 1 / constants.MASS * R @ constants.DI_DD_D @ R.T @ vel
        + forces_dist / constants.MASS
    )
    rpy_rates_dot = cs.vertcat(
        constants.DI_DD_ROLL[0] * roll
        + constants.DI_DD_ROLL[1] * droll
        + constants.DI_DD_ROLL[2] * r_cmd,
        constants.DI_DD_PITCH[0] * pitch
        + constants.DI_DD_PITCH[1] * dpitch
        + constants.DI_DD_PITCH[2] * p_cmd,
        constants.DI_DD_YAW[0] * yaw
        + constants.DI_DD_YAW[1] * dyaw
        + constants.DI_DD_YAW[2] * y_cmd,
    )
    thrust_dot = 1 / constants.DI_DD_ACC[1] * (thrust_cmd - thrust)
    theta_dot = v_theta
    states_dot = cs.vertcat(pos_dot, rpy_dot, vel_dot, rpy_rates_dot, thrust_dot, theta_dot)

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = states_dot
    model.f_impl_expr = None
    model.x = states
    model.u = inputs
    model.p = forces_dist

    return model


def create_ocp_solver(
    Tf: float, N: int, waypoints: dict[str, NDArray], verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    model = export_quadrotor_ode_model()
    ocp.model = model

    # Extract model
    x = model.x
    u = model.u
    pos = x[0:3]
    # rpy = x[3:6]
    # vel = x[6:9]
    # rpy_rates = x[9:12]
    # thrust = x[12]
    theta = x[13]
    # rpy_cmd = u[0:3]
    # thrust_cmd = u[3]
    v_theta = u[4]

    # Dimensions
    ocp.solver_options.N_horizon = N

    # Reference interpolant from the waypoints
    # Warning: Directly using bsplines on the (noninterpolated) waypoints would be
    # better, but somehow the solver doesn't converge in that case...
    lut_px = cs.interpolant("lut_px", "linear", [waypoints["time"]], waypoints["pos"][:, 0])
    lut_py = cs.interpolant("lut_py", "linear", [waypoints["time"]], waypoints["pos"][:, 1])
    lut_pz = cs.interpolant("lut_pz", "linear", [waypoints["time"]], waypoints["pos"][:, 2])
    pos_ref = cs.vertcat(lut_px(theta), lut_py(theta), lut_pz(theta))
    vel_ref = cs.vertcat(
        cs.gradient(pos_ref[0], theta),
        cs.gradient(pos_ref[1], theta),
        cs.gradient(pos_ref[2], theta),
    )
    # Also, this should be the better way of creating the LUT for the velocity
    # However, the solver finds also no solution...
    # lut_vx = cs.interpolant("lut_vx", "linear", [waypoints["time"]], waypoints["vel"][:, 0])
    # lut_vy = cs.interpolant("lut_vy", "linear", [waypoints["time"]], waypoints["vel"][:, 1])
    # lut_vz = cs.interpolant("lut_vz", "linear", [waypoints["time"]], waypoints["vel"][:, 2])
    # vel_ref = cs.vertcat(lut_vx(theta), lut_vy(theta), lut_vz(theta))
    u_ref = [0, 0, 0, constants.GRAVITY * constants.MASS, 1.0]

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # Weights
    Q_c = np.eye(3) * 50  # contour
    Q_l = np.eye(3) * 1_000  # lag
    # Q_omega = np.eye(3) * 0.5  # rpy rates velocity
    mu = 0.1  # progress
    R = np.diag(
        [
            3.0,  # rpy
            3.0,  # rpy
            3.0,  # rpy
            1.0,  # thrust
            1.0,  # v_theta
        ]
    )

    # Errors
    e = cs.vertcat(pos - pos_ref)
    normal = vel_ref
    normal = normal / cs.norm_2(normal)

    # Contouring & lag error estimation
    e_l_hat = cs.dot(normal, e) * normal
    e_c_hat = e - e_l_hat

    # Costs
    stage_cost = (
        e_c_hat.T @ Q_c @ e_c_hat
        + e_l_hat.T @ Q_l @ e_l_hat
        - mu * v_theta
        # + rpy_rates.T @ Q_omega @ rpy_rates
        + (u - u_ref).T @ R @ (u - u_ref)  # the actual inputs to the system
    )
    terminal_cost = (
        e_c_hat.T @ Q_c @ e_c_hat + e_l_hat.T @ Q_l @ e_l_hat  # + rpy_rates.T @ Q_omega @ rpy_rates
    )  # No inputs in the terminal cost!
    # Starting cost seems to make it more instable and the inital state is fixed anyway.
    # ocp.model.cost_expr_ext_cost_0 = stage_cost
    ocp.model.cost_expr_ext_cost = stage_cost
    ocp.model.cost_expr_ext_cost_e = terminal_cost

    # Set State Constraints
    # ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    # ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    # ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # Set Input Constraints
    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0, constants.THRUST_MIN * 4, 0.25])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, constants.THRUST_MAX * 4, 4.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((x.rows()))
    # ocp.dims.np = 3
    ocp.parameter_values = np.zeros(model.p.rows())

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hpipm_mode = "ROBUST"  # BALANCE, SPEED, ROBUST, SPEED_ABS
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.warm_start = True
    ocp.solver_options.qp_solver_warm_start = True
    ocp.solver_options.nlp_solver_warm_start_first_qp = True

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file=f"c_generated_code/{model.name}.json", verbose=verbose
    )

    return acados_ocp_solver, ocp


class AttitudeController(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        # MPC Settings
        self._N = config["MPC_N"]
        self._T = config["MPC_T"]
        self._dt = self._T / self._N
        self._delta_f = config["env_freq"] * self._dt
        assert np.isclose(self._delta_f % 1, 0), (
            "Controller and prediction frequency don't have a common multiple"
        )
        self._delta_f = int(self._delta_f)

        # Creating reference
        waypoints = cubic_spline_trajectory(
            config["waypoints"],
            config["desired_completion_time"],
            500,
            overhang=self._N,
            # plot=True
        )
        self._waypoints_time = waypoints["time"]
        self._waypoints_pos = waypoints["pos"]
        self._waypoints_vel = waypoints["vel"]
        self._waypoints_yaw = waypoints["yaw"]

        # Creating solver
        self._acados_ocp_solver, self._ocp = create_ocp_solver(self._T, self._N, waypoints)
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        # Controller variables
        self._t0 = time.time()
        self._theta_hat = 0
        self._theta_hat_max = config["desired_completion_time"]
        self._i_forces_dist = np.zeros(3)  # integral term
        self._last_forces_dist = np.zeros(3)
        self._forces_dist_history = deque(maxlen=2 * config["env_freq"])
        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._config = config
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The commaned euler angles and collective thrust [roll, pitch, yaw, thrust] as a numpy array.
        """
        if self._tick > self._tick_max or self._theta_hat >= self._theta_hat_max:
            self._finished = True

        # Set initial state
        self._x0 = np.concat(
            (
                obs["pos"],
                obs["rpy"],
                obs["vel"],
                obs["rpy_rates"],
                [np.sum(obs["forces_motor"])],
                [self._theta_hat],
            )
        )
        self._acados_ocp_solver.set(0, "lbx", self._x0)
        self._acados_ocp_solver.set(0, "ubx", self._x0)

        self._forces_dist_history.append(obs["forces_dist"])

        # Set disturbance over horizon
        f = [np.zeros(3)] * self._N
        if self._theta_hat > 2:
            match self._config["force_estimate_type"]:
                case "zero":
                    f = [np.zeros(3)] * self._N
                case "constant":
                    f = [obs["forces_dist"]] * self._N
                case "linear":
                    f0 = obs["forces_dist"]
                    df0 = (f0 - self._last_forces_dist) / self._config["env_freq"]
                    f, _ = pred.rollout_disturbance_force_2nd_ord(f0, df0, 0, 0, self._dt, self._N)
                case "2nd_order_system":
                    f0 = obs["forces_dist"]
                    df0 = (f0 - self._last_forces_dist) / self._config["env_freq"]
                    self._i_forces_dist += 0.001 * (f0 - self._i_forces_dist)
                    f, _ = pred.rollout_disturbance_force_2nd_ord(
                        f0, df0, 15, 37, self._dt, self._N, self._i_forces_dist
                    )
                case "LS":
                    if len(self._forces_dist_history) == self._forces_dist_history.maxlen:
                        t = time.time() - self._t0
                        f, _ = pred.disturbance_force_LS(
                            np.array((self._forces_dist_history)), t=t, step=self._delta_f
                        )
                    else:
                        f = [obs["forces_dist"]] * self._N
                case _:
                    print("[WARNING] Unknown force estimate type, using zero.")
        for i in range(self._N):
            self._acados_ocp_solver.set(i, "p", f[i])
        # print(f"{f=}")

        _ = self._acados_ocp_solver.solve()
        # if _ != 0:
        #     print(f"Status {_}: {ACADOS_STATUS[_]}")
        #     self._acados_ocp_solver.print_statistics()
        self._u0 = self._acados_ocp_solver.get(0, "u")
        # only works if prediction frequency equals MPC frequency!
        # self._theta_hat = self._acados_ocp_solver.get(1, "x")[-1]
        self._theta_hat += self._u0[-1] / self._config["env_freq"]

        self._x = [self._acados_ocp_solver.get(stage, "x") for stage in range(self._N + 1)]

        return self._u0[:4]

    def step_callback(
        self,
        action: NDArray[np.floating] | None = None,
        obs: dict[str, NDArray[np.floating]] | None = None,
        reward: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
        info: dict | None = None,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0