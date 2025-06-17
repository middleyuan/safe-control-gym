"""
Copyright © 2023 Hs293Go

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import warnings
from typing import NamedTuple

import numpy as np
import casadi as cs
import scipy.linalg as la
from scipy import optimize
import matplotlib.pyplot as plt

class Waypoint(dict):
    def __init__(
            self, time, position, velocity=None, acceleration=None, jerk=None, snap=None
    ):
        # This class is based on dict (instead of some sequential container) to support
        # the unlikely case there are 'gaps' in
        # orders of derivative specified

        self[-1] = time
        self[0] = np.asarray(position)
        if velocity is not None:
            self[1] = np.asarray(velocity)

        if acceleration is not None:
            self[2] = np.asarray(acceleration)

        if jerk is not None:
            self[3] = np.asarray(jerk)

        if snap is not None:
            self[4] = np.asarray(snap)

    @property
    def time(self):
        return self[-1]

    @property
    def position(self):
        return self[0]

    @property
    def velocity(self):
        return self[1]

    @property
    def acceleration(self):
        return self[2]


class PolynomialSize(NamedTuple):
    n_poly: int  # Number of references - 1
    n_cfs: int  # degree + 1
    dim: int


class PiecewisePolynomialTrajectory(NamedTuple):
    time_reference: np.ndarray
    durations: np.ndarray
    coefficients: np.ndarray


class RotorDragParameters(NamedTuple):
    cp: float
    dh: float
    dv: float


class QuadrotorTrajectory(NamedTuple):
    state: np.ndarray
    input: np.ndarray

    @property
    def position(self):
        return self.state[:, 0:3]

    @property
    def attitude(self):
        return self.state[:, 3:7]

    @property
    def velocity(self):
        return self.state[:, 7:10]

    @property
    def thrust(self):
        return self.input[:, 0]

    @property
    def body_rates(self):
        return self.input[:, 1:4]


def generate_trajectory(
        references,
        degree,
        *,
        idx_minimized_orders=4,
        num_continuous_orders=3,
        algorithm='closed-form',
        optimize_options=None,
        acc_limits=(-np.inf, np.inf),
):
    '''
    
    Args:
        references: List of Waypoint objectsm {-1: time, 0: position}
        degree: Degree of the polynomial
    '''
    if algorithm == "closed-form":
        print("Using closed-form solution for minimum snap, acceleration limits will be ignored")

    if degree < 2:
        raise ValueError('Polynomial degree too low')

    derivative_weights = np.zeros(degree)

    idx_minimized_orders = np.asarray(idx_minimized_orders, dtype=np.int32).ravel()
    if (idx_minimized_orders < 2).any():
        raise ValueError('Minimizing 0th- or 1st-order derivatives does not make sense')

    if (idx_minimized_orders > degree).any():
        raise ValueError(
            'Cannot minimize any derivatives whose order is higher than the degree of'
            ' the polynomial'
        )
    derivative_weights[idx_minimized_orders] = 1

    if num_continuous_orders < 3:
        raise ValueError(
            f'Constraining {num_continuous_orders} < 2 derivatives of position (velocity'
            ' and acceleration) usually ;does not make sense'
        )
    if num_continuous_orders > degree:
        raise ValueError(
            f'Cannot constrain {num_continuous_orders}-order derivatives when polynomial is'
            f' {degree}-degree only'
        )
    t_ref, refs = _parse_references(references, num_continuous_orders)

    if (t_ref < 0.0).any():
        raise ValueError('Waypoint timestamp is negative')
    durations = np.diff(t_ref).astype('float64')
    if (durations <= 1e-8).any():
        raise ValueError(
            'The time duration for transiting between waypoints is too small'
        )

    poly_dim = PolynomialSize(
        n_poly=refs.shape[0] - 1, n_cfs=degree + 1, dim=refs.shape[2]
    )
    '''
    n_poly: num of segments = num of references - 1
    n_cfs: num of coefficients = degree + 1
    dim: dimension of the trajectory = num of continuous orders
    '''

    if algorithm == "constrained":
        solver = _solve_constrained
        polys = solver(
            refs,
            durations,
            poly_dim,
            derivative_weights,
            num_continuous_orders,
            optimize_options,
            acc_limits,
        )
    elif algorithm == "closed-form":
        solver = _solve_closed_form
        polys = solver(
            refs,
            durations,
            poly_dim,
            derivative_weights,
            num_continuous_orders,
            optimize_options,
        )
    else:
        raise ValueError('Unrecognized algorithm')
    polys = solver(
        refs,  # parsed references
        durations,  # time intervals between waypoints
        poly_dim,  # PolynomialSize object
        derivative_weights,  # weights for derivatives (e.g. minimum snap)
        num_continuous_orders,
        optimize_options,
    )
    return PiecewisePolynomialTrajectory(t_ref, durations, polys)


def _parse_references(references, r_cts):
    len_traj = len(references)

    t_ref = []
    trajectory_ref = []
    for idx, it in enumerate(references):
        ref = []
        t_ref.append(it.time)
        ref.append(it.position)
        dim = len(it.position)

        if any(k > r_cts for k in it.keys()):
            warnings.warn('Too many derivatives specified', stacklevel=2)
        # velocity/acceleration are constrained to zero at initial or terminal points;
        # Otherwise they are unconstrained as signalled by nans
        # Higher-order derivatives are always unconstrained unless specified
        for r in range(1, r_cts):
            if r < 3 and idx in (0, len_traj - 1):
                placeholder = np.zeros(dim)
            else:
                placeholder = np.full([dim], np.nan)
            ref.append(it.get(r, placeholder))
        trajectory_ref.append(np.array(ref))
    trajectory_ref = np.asarray(trajectory_ref)
    t_ref = np.asarray(t_ref)
    return t_ref, trajectory_ref


def _solve_closed_form(
        refs,
        durations,
        poly_dim,
        derivative_weights,
        r_cts,
        optimize_options,
):
    if r_cts < 3:
        raise ValueError(
            'Trajectory must be continuous up to the 2nd order (acceleration)'
        )

    if optimize_options is not None:
        warnings.warn(
            'Solving the trajectory generation problem in closed form.'
            'Optimizer options will be ignored',
            stacklevel=2,
        )
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    Q = np.zeros((n_vars, n_vars))
    for r, c_r in enumerate(derivative_weights):
        Q += c_r * la.block_diag(*_compute_Q(poly_dim.n_cfs, r, durations))

    poly_coeffs = np.zeros(poly_dim)
    A = np.zeros((r_cts * 2 * poly_dim.n_poly, poly_dim.n_cfs * poly_dim.n_poly))
    for i in range(poly_dim.n_poly):
        s = np.s_[poly_dim.n_cfs * i: poly_dim.n_cfs * (i + 1)]
        for r in range(r_cts):
            A[r_cts * 2 * i + r, s] = (
                    _compute_tvec(poly_dim.n_cfs, r, 0) / durations[i] ** r
            )
            A[r_cts * (2 * i + 1) + r, s] = (
                    _compute_tvec(poly_dim.n_cfs, r, 1) / durations[i] ** r
            )

    M = np.zeros((poly_dim.n_poly * 2 * r_cts, r_cts * (poly_dim.n_poly + 1)))
    for i in range(poly_dim.n_poly):
        s1 = np.s_[2 * r_cts * i: 2 * r_cts * (i + 1)]
        s2 = np.s_[r_cts * i: r_cts * (i + 2)]
        M[s1, s2] = np.eye(2 * r_cts)

    num_d = r_cts * (poly_dim.n_poly + 1)

    # compute C
    C = np.eye(num_d)
    # fix all pos(poly_dim.n_poly+1) + start va(2) +  va(2)
    fix_idx = np.flatnonzero(np.all(~np.isnan(refs), axis=2).ravel())
    free_idx = np.setdiff1d(np.arange(num_d), fix_idx)
    C = np.hstack([C[:, fix_idx], C[:, free_idx]])

    res = la.lstsq(A, M @ C)
    assert res is not None
    AiMC = res[0]
    R = AiMC.T @ Q @ AiMC

    n_fix = fix_idx.size
    Rpp = R[n_fix:, n_fix:]
    Rfp = R[:n_fix, n_fix:]

    for d in range(poly_dim.dim):
        ref = refs[..., d]
        df = ref.ravel()[fix_idx]
        dp = -la.solve(Rpp, Rfp.T @ df)

        coeffs = np.reshape(AiMC @ np.concatenate([df, dp]), poly_dim[0:2])
        poly_coeffs[:, :, d] = (1.0 / durations[..., None]) ** np.arange(
            0, poly_dim.n_cfs
        ) * coeffs

    return poly_coeffs


def _solve_constrained(
        refs,
        durations,
        poly_dim,
        derivative_weights,
        r_cts,
        optimize_options,
        acc_limits=(-np.inf, np.inf),
):
    '''
    # reminder: 
    n_poly: num of segments = num of references - 1
    n_cfs: num of coefficients = degree + 1
    dim: dimension of the trajectory (x y z)
    '''
    '''
    refs: np.ndarray, shape=(n_refs, r_cts, dim)
    durations: np.ndarray, shape=(n_refs-1,)
    poly_dim: PolynomialSize object (n_poly, n_cfs, dim)
    derivative_weights: np.ndarray, shape=(n_cfs,)
    r_cts: num of continuous orders
    optimize_options: dict
    acc_limits: tuple, (min, max) acceleration limits
    '''
    opts = {"method": "SLSQP", "tol": 1e-10}
    if optimize_options is not None:
        opts.update(optimize_options)
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    Q = np.zeros((n_vars, n_vars))
    for r, c_r in enumerate(derivative_weights):
        Q += c_r * la.block_diag(*_compute_Q(poly_dim.n_cfs, r, durations))

    poly_coeffs = np.zeros(poly_dim)
    # for the continuity at the waypoints
    Aeq_1, beq_1 = _compute_continuity_constraints(poly_dim, durations, r_cts)
    for d in range(poly_dim.dim):  # for each dimension (x y z)
        Aeq_0, beq_0 = _compute_dynamical_constraints(
            poly_dim, refs[:, :, d], durations
        )

        Aeq = np.vstack([Aeq_0, Aeq_1])
        beq = np.concatenate([beq_0, beq_1])
        # constr = optimize.LinearConstraint(Aeq, beq, beq)  # type: ignore

        Aineq, bineq_ub, bineq_lb = _compute_acceleration_constraints(
            poly_dim, refs[:, :, d], durations, acc_limits
        )
        Aeq = np.vstack([Aeq, Aineq])
        bineq_ub = np.concatenate([beq, bineq_ub])
        bineq_lb = np.concatenate([beq, bineq_lb])
        constr = optimize.LinearConstraint(Aeq, bineq_lb, bineq_ub)  # type: ignore

        soln = optimize.minimize(
            lambda x: (x @ Q @ x) / 2,
            np.zeros(n_vars),
            constraints=constr,
            jac=lambda x: Q @ x,
            **opts,
        )
        coeffs = soln.x.reshape(poly_dim[0:2])
        poly_coeffs[:, :, d] = (1.0 / durations[..., None]) ** np.arange(
            0, poly_dim.n_cfs
        ) * coeffs
    return poly_coeffs


def _compute_continuity_constraints(poly_dim, durations, r_cts):
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    Aeq = np.zeros(((poly_dim.n_poly - 1) * r_cts, n_vars))
    beq = np.zeros((poly_dim.n_poly - 1) * r_cts)
    for i in range(poly_dim.n_poly - 1):
        s = np.s_[poly_dim.n_cfs * i: poly_dim.n_cfs * (i + 2)]
        for r in range(r_cts):
            tvec_l = _compute_tvec(poly_dim.n_cfs, r, 1) / durations[i] ** r
            tvec_r = _compute_tvec(poly_dim.n_cfs, r, 0) / durations[i + 1] ** r
            Aeq[r_cts * i + r, s] = np.concatenate([tvec_l, -tvec_r])
    return Aeq, beq


def _compute_dynamical_constraints(poly_dim, refs, durations):
    '''
    compute the waypoints constraints for the trajectory (position, velocity, acceleration, etc.)

    Args:
        poly_dim: PolynomialSize object (n_poly, n_cfs, dim)
        refs: np.ndarray (waypoints, r_cts, dim(x,y,z)
        durations: np.ndarray, shape=(n_refs-1,)
    '''
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    n_constrain_orders = np.count_nonzero(~np.isnan(refs), axis=1)
    Aeq = np.zeros((n_constrain_orders.sum(), n_vars))
    beq = np.zeros(n_constrain_orders.sum())

    row_its = np.concatenate([[0], n_constrain_orders.cumsum()])
    for i in range(poly_dim.n_poly + 1):
        idx, tau = (i - 1, 1.0) if i == poly_dim.n_poly else (i, 0.0)
        s = np.s_[poly_dim.n_cfs * idx: poly_dim.n_cfs * (1 + idx)]
        for r in range(n_constrain_orders[i]):
            Aeq[row_its[i] + r, s] = (
                    _compute_tvec(poly_dim.n_cfs, r, tau) / durations[idx] ** r
            )
            beq[row_its[i] + r] = refs[i, r]
    return Aeq, beq


def _compute_acceleration_constraints(poly_dim, refs, durations, acc_limits):
    '''
    compute the inequality constraint to limit the acceleration at waypoints
    '''
    acc_min, acc_max = acc_limits
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    # n_constrain_orders = 3 
    Aineq = np.zeros((poly_dim.n_poly, n_vars))
    bineq_ub = np.zeros(poly_dim.n_poly)
    bineq_lb = np.zeros(poly_dim.n_poly)

    for i in range(poly_dim.n_poly):
        s = np.s_[poly_dim.n_cfs * i: poly_dim.n_cfs * (i + 1)]
        Aineq[i, s] = _compute_tvec(poly_dim.n_cfs, 2, 1) / durations[i] ** 2
        bineq_ub[i] = acc_max
        bineq_lb[i] = acc_min
    return Aineq, bineq_ub, bineq_lb


def _compute_Q(n_cfs, r, tau):  # pylint: disable=C0103
    Q = np.zeros((len(tau), n_cfs, n_cfs))

    i, l = np.meshgrid(*[np.arange(r, n_cfs)] * 2, sparse=True)  # NOQA
    m_seq = np.arange(0, r)[:, None, None]
    k = -2 * r + 1
    Q[:, i, l] = (
            np.prod((i - m_seq) * (l - m_seq), axis=0) / (k + i + l) * tau[:, None, None] ** k
    )
    return Q


def _compute_tvec(n_cfs, r, tau):
    '''
    Compute the vector of monomials for a given order and time
    
    Args:
        n_cfs: int, number of coefficients = degree + 1
        r: int, order of the derivative (0 for position, 1 for velocity, etc.)
        tau: float, time (0 for start, 1 for end)
    '''
    tvec = np.zeros(n_cfs)
    n_seq = np.arange(r, n_cfs)
    r_seq = np.arange(0, r)[:, None]
    tvec[n_seq] = np.prod(n_seq - r_seq, axis=0) * tau ** (n_seq - r)
    return tvec


def compute_trajectory_derivatives(polys, t_sample, order):
    t_sample = np.asarray(t_sample)
    t_ref, _, coeffs = polys

    _, n_cfs, dim = coeffs.shape
    if order < 0:
        raise ValueError('Negative derivative order')
    if order > n_cfs - 1:
        raise ValueError(f'Kinematic order {order} > polynomial degree {n_cfs - 1}')
    len_traj = t_sample.size
    trajectory_derivatives = np.zeros((order, len_traj, dim), dtype=np.float64)

    def find_piece(t):
        if not t_ref[0] <= t <= t_ref[-1]:
            warnings.warn('Query point is outside of bounds. Clamping', stacklevel=2)
            t = np.clip(t, t_ref[0], t_ref[-1])
        idx = np.flatnonzero(t >= t_ref[:-1])[-1]
        tau = t - t_ref[idx]
        return coeffs[idx, ...], tau

    for r in range(order):
        for k, t in enumerate(t_sample):
            piece, segment_time = find_piece(t)
            trajectory_derivatives[r, k, :] = _nd_polyvals(piece, segment_time, r)
    return trajectory_derivatives


def _nd_polyvals(coeffs, time, r):
    n_cfs = coeffs.shape[0]  # Coefficients per-piece
    if r == 0:
        return (time ** np.arange(0, n_cfs)) @ coeffs
    n_seq = np.arange(r, n_cfs, dtype=np.int64)
    r_seq = np.arange(0, r, dtype=np.int64)
    return time ** (n_seq - r) @ (
            np.prod(n_seq[None, :] - r_seq[:, None], axis=0)[..., None] * coeffs[n_seq, :]
    )


def _distance_to_point(point1, point2):
    return (point1 - point2).T @ (point1 - point2)


def _distance_to_line(start, end, point):
    cross = cs.cross(start - end, start - point)
    string = start - end
    return cs.sqrt((cross.T @ cross) / (string.T @ string))

def _plot_trajectory(pos_ref, waypoints=None, strings=None, save_path=None):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 4))
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], 'b', label="Position Trajectory")
    if waypoints is not None:
        for i, waypoint in enumerate(waypoints):
            position = waypoint.position
            ax.plot(
                position[0],
                position[1],
                position[2],
                "go",
                label="Position Waypoints" if i == 0 else ""
            )
    if strings is not None:
        for i, string in enumerate(strings):
            start = string['start']
            end = string['end']
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r', label="Strings"  if i == 0 else "")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    try:
        fig.savefig(save_path)
        print(f"Trajectory plot saved to {save_path}")
    except:
        pass
    plt.show()
    plt.close(fig)

def _plot_xyz_kinematics(pos_ref, vel_ref=None, acc_ref=None, speed_ref=None, waypoints=None, strings=None, save_path=None):
    if pos_ref is None:
        print("Missing position reference data for kinematic plotting.")
        return

    time_ref = np.linspace(0, len(pos_ref) - 1, len(pos_ref))  # Assumes uniform time steps
    axis_labels = ['X', 'Y', 'Z']

    # Plot position, velocity, and acceleration for each axis
    for i in range(3):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(time_ref, pos_ref[:, i], 'b-', linewidth=1.5, label=f'Position {axis_labels[i]}')

        if vel_ref is not None:
            ax.plot(time_ref, vel_ref[:, i], 'g-', linewidth=1.5, label=f'Velocity {axis_labels[i]}')

        if acc_ref is not None:
            ax.plot(time_ref, acc_ref[:, i], 'm-', linewidth=1.5, label=f'Acceleration {axis_labels[i]}')

        ax.set_xlabel("Time (arbitrary units)")
        ax.set_ylabel(f"{axis_labels[i]}-axis")
        ax.set_title(f"{axis_labels[i]}-Axis Kinematics")
        ax.legend(loc="upper right")
        ax.grid(True)
        fig.tight_layout()

        if save_path is not None:
            save_axis_path = f"{save_path}_{axis_labels[i].lower()}.png"
            try:
                fig.savefig(save_axis_path)
                print(f"Kinematics plot saved to {save_axis_path}")
            except Exception as e:
                print(f"Failed to save plot {save_axis_path}: {e}")

        plt.show()
        plt.close(fig)

    # Plot speed magnitude on separate plot
    if speed_ref is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(time_ref, speed_ref, 'r-', linewidth=1.5, label='Speed (Magnitude)')
        ax.set_xlabel("Time (arbitrary units)")
        ax.set_ylabel("Speed")
        ax.set_title("Speed Magnitude vs Time")
        ax.legend(loc="upper right")
        ax.grid(True)
        fig.tight_layout()

        if save_path is not None:
            save_speed_path = f"{save_path}_speed.png"
            try:
                fig.savefig(save_speed_path)
                print(f"Speed plot saved to {save_speed_path}")
            except Exception as e:
                print(f"Failed to save plot {save_speed_path}: {e}")

        plt.show()
        plt.close(fig)

class TrajectoryPlanner:
    def __init__(self, waypoint_list, string_list, N=30):
        self.waypoint_list = waypoint_list
        self.string_list = string_list
        self.T = waypoint_list[-1]['time']  # Trajectory length in time
        self.N = N  # number of waypoints
        self.dt = self.T / N
        self.string_discrete_point = 10

        # waypoints
        length_w = len(waypoint_list)
        self.start_loc = waypoint_list[0]['position']
        self.end_loc = waypoint_list[length_w - 1]['position']
        pos_init = np.array([]).reshape(0, 6)
        for i in range(length_w - 1):
            dt = waypoint_list[i + 1]['time'] - waypoint_list[i]['time']
            sub_waypoints = np.linspace(waypoint_list[i]['position'] + [0., 0., 0.],
                                        waypoint_list[i + 1]['position'] + [0., 0., 0.],
                                        round(dt * N / self.T))
            pos_init = np.concatenate((pos_init, sub_waypoints), axis=0)
        pos_init = np.concatenate((pos_init,
                                   np.array(waypoint_list[-1]['position'] + [0., 0., 0.])[None, :]), axis=0)
        self.dynamics_fn()
        self.traj_solver = self.traj_optimizer()
        x0 = np.concatenate((np.zeros((self.N * 3, 1)),
                             pos_init.reshape(-1, 1),
                             np.zeros(((self.N + 1) * len(self.string_list) * self.string_discrete_point, 1))), axis=0)

        soln = self.traj_solver(x0=x0, p=[], lbg=self.lbg, ubg=self.ubg)
        ref = soln['x'].full()
        if not self.traj_solver.stats()['success']:
            print('Trajectory planner failed')
        # act_ref = ref[:self.N * 3, :].reshape(self.N, 3)
        state_ref = ref[self.N * 3: self.N * 3 + 6 * (self.N + 1), :].reshape(self.N + 1, 6)
        pos_ref = state_ref[:, :3].copy()
        # vel_ref = state_ref[:, 3:].copy()

        self.waypoints = []
        for i in range(pos_ref.shape[0]):
            self.waypoints.append(
                Waypoint(
                    time=i * self.dt,
                    position=pos_ref[i, :],
                    # velocity=vel_ref[i, :]
                )
            )

    def dynamics_fn(self):
        x = cs.MX.sym('x', 6)
        u = cs.MX.sym('u', 3)

        A = np.array([[1., 0., 0., self.dt, 0., 0.], [0., 1., 0., 0., self.dt, 0.], [0., 0., 1., 0., 0., self.dt],
                      [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]])
        B = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                      [self.dt, 0., 0.], [0., self.dt, 0.], [0., 0., self.dt]])

        self.dyn = cs.Function('dynamics', [x, u], [A @ x + B @ u])

    def traj_optimizer(self):
        X = cs.MX.sym('X', 6, self.N + 1)
        U = cs.MX.sym('U', 3, self.N)
        Sigma = cs.MX.sym('Sigma', len(self.string_list) * self.string_discrete_point, self.N + 1)

        opt_vars = cs.vertcat(
            cs.reshape(U, -1, 1),
            cs.reshape(X, -1, 1),
            cs.reshape(Sigma, -1, 1)
        )
        # acceleration limits
        lb = np.array([-10.0, -10.0, -10.0])
        ub = np.array([10.0, 10.0, 10.0])

        cost = 0
        g, h = [], []
        g.append(X[:3, 0] - np.array(self.start_loc))
        g.append(X[3:, 0])
        g.append(X[:3, -1] - np.array(self.end_loc))
        g.append(X[3:, -1])
        for i in range(self.N):
            cost += U[:, i].T @ U[:, i]
            h.append(U[:, i] - ub)
            h.append(lb - U[:, i])
            x_next = self.dyn(X[:, i], U[:, i])
            g.append(x_next - X[:, i + 1])

            for j, string in enumerate(self.string_list):
                for k, point in enumerate(np.linspace(string['start'], string['end'], self.string_discrete_point)):
                    d = _distance_to_point(point, X[:3, i])
                    h.append(0.5 - d - Sigma[j * k, i])
                    h.append(-Sigma[j * k, i])
            cost += 1e2 * Sigma[:, i].T @ Sigma[:, i]
        
        for wp in self.waypoint_list[1:-1]:  # skip start and endAdd commentMore actions
            t_idx = int(wp['time'] / self.dt)  # convert time to index
            pos = X[:3, t_idx]  # predicted position at that time
            desired = cs.vertcat(*wp['position'])  # desired waypoint
            cost += 5 * cs.sumsqr(pos - desired)  # soft penalty

        G = cs.vertcat(*g)
        H = cs.vertcat(*h)
        G_con = cs.vertcat(*g, *h)
        self.lbg = cs.vertcat(*([0] * G.shape[0] + [-np.inf] * H.shape[0]))
        self.ubg = cs.vertcat(*([0] * G.shape[0] + [0] * H.shape[0]))

        opts_setting = {
            'print_time': 0,
            'expand': True,
            'fatrop.max_iter': 200,
            'fatrop.print_level': 0,
            'fatrop.acceptable_tol': 1e-5,
        }
        nlp_prob = {
            'f': cost,
            'x': opt_vars,
            'p': [],
            'g': G_con,
        }
        traj_solver = cs.nlpsol('traj_solver', 'fatrop', nlp_prob, opts_setting)
        return traj_solver
