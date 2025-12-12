"""Plotting script."""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp
import toml
import torch
from array_api_compat import array_namespace

from utils import wavelet_decompose  # , VMD

if TYPE_CHECKING:
    from typing import Callable

    from numpy.typing import NDArray

with open("drones.toml", "r") as f:
    drones = toml.load(f)
config = drones["drone0"]
drone_name = f"cf{config['id']}"
experiment = config["experiment"]  # wind, slung_load
controller = config[
    "controller"
]  # "basic_mpc", "basic_mpcc", "force_mpcc", "force_mpc_constant", "force_mpc_NN"
controller_verif = "force_mpc_constant"
MPC_N = config["MPC_N"]

try:
    with open(f"data/{experiment}/wind_field_rbfs.pkl", "rb") as f:
        rbfs = pickle.load(f)
    rbf_fx = rbfs["rbf_fx"]
    rbf_fy = rbfs["rbf_fy"]
except Exception:
    print("[ERROR] Couldnt load rbfs")

wavelets = ["A7", "D7", "D6"]
# models = {w: torch.load(f"full_model_{w}.pth", weights_only=False) for w in wavelets}
# model = torch.load("full_model_.pth", weights_only=False)
try:
    model_VMD = torch.load("full_model_VMD.pth", weights_only=False)
except Exception:
    print("Couldnt load VMD model")

# Data for LS approach
runs = []  # list of tuples (t, f)
decompose = False  # whether to decompose the forces or not
controllers = ["force_mpc_constant", "force_mpc_LS"]
if "mpcc" in controller:
    controllers = ["force_mpcc_constant", "force_mpcc_LS"]
for c in controllers:
    data_e_RMSE_controller = []
    for e in ["wind_4x3", "payload"]:  # , "wind_payload" ["wind_4x3", "payload"], ["wind_slow"]
        for filename in os.listdir(f"data/{e}"):
            file_path = os.path.join(f"data/{e}", filename)
            if "003" in file_path or "004" in file_path or "005" in file_path or "006" in file_path:
                continue
            if c + "_run" in file_path:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                # Extract information
                t = data["t"] - data["t"][0]
                obs = data["obs"]
                start_idx = np.searchsorted(t, 1.0)
                stop_idx = -1  # np.searchsorted(t, 6.0)

                t = t[start_idx:stop_idx]
                f = obs["forces_dist"][start_idx:stop_idx]

                if decompose:
                    f_decomp = wavelet_decompose(f, level=7)
                    for w in ["A7", "D7", "D6", "D5"]:  # , "D4"
                        # Check for orthogonality
                        is_orthogonal = True
                        f_new = f_decomp[w]
                        # for _, f_existing in runs:
                        #     min_len = min(len(f_new), len(f_existing))
                        #     dot_product = np.dot(
                        #         f_new[:min_len].flatten(), f_existing[:min_len].flatten()
                        #     )
                        #     norm_product = np.linalg.norm(
                        #         f_new[:min_len].flatten()
                        #     ) * np.linalg.norm(f_existing[:min_len].flatten())
                        #     if norm_product > 1e-1 and abs(dot_product / norm_product) > 1e-2:
                        #         is_orthogonal = False
                        #         break
                        if is_orthogonal:
                            runs.append((t[: len(f_new)], f_new))
                else:
                    runs.append((t, f))  # regular runs with noise!

                # runs.append((t, np.roll(f, 16, axis=0)))
                # runs.append((t, np.roll(f, -16, axis=0)))

print(f"[INFO] Loaded {len(runs)} runs for LS")

# Add constant runs
t = np.arange(0, 60, 1 / 200)
f = np.zeros((len(t), 3))
f[:, 0] = 0.05
runs.append((t, f))
f = np.zeros((len(t), 3))
f[:, 1] = 0.05
runs.append((t, f))
f = np.zeros((len(t), 3))
f[:, 2] = 0.05
runs.append((t, f))


def disturbance_force_map(pos: NDArray) -> NDArray:
    """Returns the expected disturbance force (x & y) for the given position (x & y)."""
    assert pos.shape[-1] == 2, "pos must only contain x and y"
    if pos.ndim == 1:
        fx = rbf_fx([pos])
        fy = rbf_fy([pos])
        return np.array([fx[0], fy[0]])
    elif pos.ndim == 2:
        fx = rbf_fx(pos)
        fy = rbf_fy(pos)
        return np.stack([fx, fy]).T
    else:
        raise NotImplementedError("pos array must be 1D (single pos) or 2D (batched)!")


def disturbance_force_LS(
    force_history: NDArray, t: float, step: int = 1
) -> tuple[NDArray, NDArray]:
    """TODO."""
    input_len = len(force_history)
    Y = force_history.flatten()

    X = []
    X_pred = []

    for r, (t_run, f_run) in enumerate(runs):
        run_idx = (
            np.searchsorted(t_run, t) + step
        )  # the +step is because we want to predict the next force
        if run_idx - input_len < 0 or run_idx + MPC_N * step >= len(f_run):
            continue  # Outside of bounds => Don't use dataset
        X.append(f_run[run_idx - input_len : run_idx, :].flatten())
        X_pred.append(f_run[run_idx : run_idx + MPC_N * step, :].flatten())
    if len(X) == 0:
        # If not dataset has data available, simply return a constant prediction
        return np.array([force_history[-1]] * MPC_N), np.zeros((len(runs)))

    X = np.array(X).T
    X_pred = np.array(X_pred).T

    reg = np.eye(X.shape[1]) * 1e-3  # regularization
    reg[-3:, -3:] = 0.0  # don't regularize the last 3 dimensions (constant offsets)
    theta = np.linalg.inv(X.T @ X + reg) @ X.T @ Y

    f_pred = (X_pred @ theta).reshape((-1, 3))[::step]
    return f_pred, theta


def disturbance_force_NN(force_history: NDArray, step: int = 1) -> NDArray:
    """TODO."""
    decomposition = wavelet_decompose(force_history, level=7)
    f_predictions = []
    for dim in range(3):
        # f_hist = {w: torch.tensor(decomposition[w][::step, dim]).unsqueeze(0) for w in wavelets}
        # f_pred = 0
        # with torch.no_grad():
        #     for w in wavelets:
        #         f_pred += models[w](f_hist[w]).squeeze().numpy()
        f_hist = torch.tensor(force_history[::step, dim]).unsqueeze(0)
        with torch.no_grad():
            f_pred = model(f_hist).squeeze().numpy()
        f_predictions.append(f_pred)
    return np.array(f_predictions).T


def disturbance_force_NN_VMD(force_history: NDArray, step: int = 1) -> NDArray:
    """TODO."""
    alpha = 100_000  # bandwidth constraint
    tau = 0.0  # noise-tolerance (no strict fidelity enforcement)
    K = 8  # modes
    K_used = 3  # only use the lowest frequency modes for recomposition
    DC = 1  # DC part
    init = 2  # initialize omegas: as zero (0), uniformly (1), random (2)
    tol = 1e-7
    omega_fixed = np.array(
        [0.0, 0.00114381, 0.00283021, 0.00555261, 0.01038964, 0.01315736, 0.02059959, 0.13352584]
    )

    f_predictions = []
    for dim in range(3):
        f_simple, _, _ = VMD(force_history[:, dim], alpha, tau, K, DC, init, tol, omega_fixed)
        f_simple = np.sum(f_simple[:K_used], axis=0)
        f_hist = torch.tensor(f_simple[::step], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            f_pred = model_VMD(f_hist).squeeze().numpy()
        f_predictions.append(f_pred)
    return np.array(f_predictions).T


def disturbance_force_time(i: int, N: int, step: int = 1) -> NDArray:
    """TODO."""
    return disturbance_force_prev[i : i + N : step]


def rollout_disturbance_force_constant(
    f0: NDArray, df0: NDArray, ddf0: NDArray, dt: float, N: int
) -> NDArray:
    """Assumes the force has either constant force, constant velocity or constant acceleration."""
    dim = f0.shape[0]

    # Initialize rollout
    f = np.zeros((N + 1, dim))
    df = np.zeros((N + 1, dim))
    f[0] = f0
    df[0] = df0

    for i in range(N):
        f[i + 1] = f[i] + df[i] * dt
        df[i + 1] = df[i] + ddf0 * dt

    return f


def rollout_disturbance_force_1st_ord(
    f0: NDArray, k: float | NDArray, dt: float, N: int
) -> NDArray:
    """Numerically stable rollout of a first-order disturbance force system.

    System: ẋ = -k * x, where x ≡ force f

    Args:
        f0: Initial force, shape (dim,)
        k: Decay coefficient (scalar or per dimension)
        dt: Time step
        N: Horizon (returns N+1 states)

    Returns:
        forces: Array of shape (N+1, dim)
    """
    dim = f0.shape[0]

    # Continuous-time system matrix A_c for x
    A_c = -k * np.eye(dim)  # handles scalar or array multiplication

    # Discretize: A_d = exp(A_c * dt)
    A_d = sp.linalg.expm(A_c * dt)

    # Initialize rollout
    f = np.zeros((N + 1, dim))
    f[0] = f0

    for i in range(N):
        f[i + 1] = A_d @ f[i]

    return f


def rollout_disturbance_force_2nd_ord(
    f0: NDArray,
    df0: NDArray,
    k1: float | NDArray,
    k2: float | NDArray,
    dt: float,
    N: int,
    f_steady: NDArray | float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Numerically stable rollout of a second-order disturbance force system.

    System: ẍ = -k1 * ẋ - k2 * x, where x ≡ force f

    Args:
        f0: Initial force, shape (dim,)
        df0: Initial force derivative, shape (dim,)
        k1: Damping-like coefficient (scalar or per dimension)
        k2: Stiffness-like coefficient (scalar or per dimension)
        dt: Time step
        N: Horizon (returns N+1 states)
        f_steady: Steady state force to converge to

    Returns:
        forces: Array of shape (N+1, dim)
        force_derivatives: Array of shape (N+1, dim)
    """
    xp = array_namespace(f0)
    dim = f0.shape[0]
    state_dim = 2 * dim

    # Continuous-time system matrix A_c for [f, df]
    A_c = xp.zeros((state_dim, state_dim))

    # df/dt = df
    A_c[0:dim, dim : 2 * dim] = xp.eye(dim)

    # ddf/dt = -k1 * df - k2 * f
    A_c[dim : 2 * dim, 0:dim] = -k2 * xp.eye(dim)
    A_c[dim : 2 * dim, dim : 2 * dim] = -k1 * xp.eye(dim)

    # Discretize
    A_d = sp.linalg.expm(A_c * dt)

    # Initialize state
    s = xp.zeros((N + 1, state_dim))
    s[0, 0:dim] = f0
    s[0, dim : 2 * dim] = df0

    # Create steady state
    s_steady = xp.zeros((state_dim))
    s_steady[0:dim] = xp.ones((dim)) * f_steady

    # Rollout
    for k in range(N):
        s[k + 1] = A_d @ (s[k] - s_steady) + s_steady

    return s[:, 0:dim], s[:, dim : 2 * dim]


def fit_force_field_2nd_ord(
    pos_history: NDArray, force_history: NDArray
) -> Callable[[NDArray], NDArray]:
    """TODO."""
    force_history_norm = np.linalg.norm(force_history[:, :2], axis=-1)
    x0, y0 = pos_history[-1, 0], pos_history[-1, 1]

    # Check if signs have changed -> if so, no prediction
    epsilon = 1e-3
    if not (
        np.all(force_history[:, 0] > epsilon) or np.all(force_history[:, 0] < -epsilon)
    ) or not (np.all(force_history[:, 1] > epsilon) or np.all(force_history[:, 1] < -epsilon)):

        def fun(pos: NDArray) -> NDArray:
            return np.zeros_like(pos)

        return fun
    fx_mean = np.mean(force_history[:, 0])
    fy_mean = np.mean(force_history[:, 1])
    angle = np.arctan2(fy_mean, fx_mean)
    # sign = np.sign(force_history[0, :2])
    q = (pos_history[:, 0] - x0) * np.cos(angle) + (pos_history[:, 1] - y0) * np.sin(angle)

    # Fit
    X = np.array([q**2, q, np.ones_like(q)]).T
    Y = np.log(force_history_norm)
    abc = sp.optimize.lsq_linear(X, Y, ([-np.inf, -50, -np.inf], [-1e-3, 50, np.inf])).x

    def fun(pos: NDArray) -> NDArray:
        q_rollout = (pos[..., 0] - x0) * np.cos(angle) + (pos[..., 1] - y0) * np.sin(angle)
        f_rollout_norm = np.exp(
            abc[0] * q_rollout**2 + abc[1] * q_rollout**1 + abc[2] * q_rollout**0
        )
        f_rollout = np.array(
            [np.cos(angle) * f_rollout_norm, np.sin(angle) * f_rollout_norm, 0 * f_rollout_norm]
        ).T
        return f_rollout

    return fun


class EKF1D:
    def __init__(self, dt, x_integral, process_noise=1e-4, meas_noise=2e-4):
        self.dt = dt
        self.x_integral = x_integral

        # State: [x, x_dot, d, k]
        self.z = np.array([0.0, 0.0, 0.1, 0.1])
        self.P = np.eye(4) * 0.1

        self.Q = np.diag([1e-8, 1e-5, process_noise, process_noise])  # process noise
        self.R = np.array([[meas_noise]])  # measurement noise

    def f(self, z):
        x, xdot, d, k = z
        xddot = -d * xdot - k * (x - self.x_integral)

        x_next = x + xdot * self.dt
        xdot_next = xdot + xddot * self.dt
        return np.array([x_next, xdot_next, d, k])

    def F_jacobian(self, z):
        x, xdot, d, k = z
        J = np.eye(4)
        J[0, 1] = self.dt
        J[1, 1] = 1 - d * self.dt
        J[1, 2] = -xdot * self.dt
        J[1, 3] = -(x - self.x_integral) * self.dt
        return J

    def h(self, z):
        return np.array([z[0]])  # only observe x

    def H_jacobian(self, z):
        return np.array([[1.0, 0.0, 0.0, 0.0]])

    def predict(self):
        self.z = self.f(self.z)
        F = self.F_jacobian(self.z)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, x_meas):
        H = self.H_jacobian(self.z)
        y = np.array([x_meas]) - self.h(self.z)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.z = self.z + (K @ y).flatten()
        self.P = (np.eye(4) - K @ H) @ self.P

    def step(self, x_meas):
        self.predict()
        self.update(x_meas)
        return self.z.copy()


class EKF1DAdaptiveIntegral:
    def __init__(self, dt, process_noise=1e-4, meas_noise=2e-4, x_integral_rate=0.001):
        self.dt = dt
        self.alpha = x_integral_rate  # convergence rate of x_integral

        # State: [x, x_dot, d, k, x_integral]
        self.z = np.array([0.0, 0.0, 0.1, 0.1, 0.0])
        self.P = np.eye(5) * 0.1

        self.Q = np.diag([1e-5, 1e-5, process_noise, process_noise, 1e-200])
        self.R = np.array([[meas_noise]])

    def f(self, z):
        x, xdot, d, k, x_int = z
        xddot = -d * xdot - k * (x - x_int)

        x_next = x + xdot * self.dt
        xdot_next = xdot + xddot * self.dt
        x_int_next = x_int + self.alpha * (x - x_int) * self.dt

        return np.array([x_next, xdot_next, d, k, x_int_next])

    def F_jacobian(self, z):
        x, xdot, d, k, x_int = z
        J = np.eye(5)

        # ∂x_next / ∂xdot
        J[0, 1] = self.dt

        # ∂xdot_next / ∂xdot
        J[1, 1] = 1 - d * self.dt
        # ∂xdot_next / ∂d
        J[1, 2] = -xdot * self.dt
        # ∂xdot_next / ∂k
        J[1, 3] = -(x - x_int) * self.dt
        # ∂xdot_next / ∂x
        J[1, 0] = -k * self.dt
        # ∂xdot_next / ∂x_integral
        J[1, 4] = k * self.dt

        # ∂x_integral_next / ∂x
        J[4, 0] = self.alpha * self.dt
        # ∂x_integral_next / ∂x_integral
        J[4, 4] = 1 - self.alpha * self.dt

        return J

    def h(self, z):
        return np.array([z[0]])  # observe x

    def H_jacobian(self, z):
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])

    def predict(self):
        F = self.F_jacobian(self.z)
        self.z = self.f(self.z)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, x_meas):
        H = self.H_jacobian(self.z)
        y = np.array([x_meas]) - self.h(self.z)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.z = self.z + (K @ y).flatten()
        self.P = (np.eye(5) - K @ H) @ self.P

    def step(self, x_meas):
        self.predict()
        self.update(x_meas)
        return self.z.copy()