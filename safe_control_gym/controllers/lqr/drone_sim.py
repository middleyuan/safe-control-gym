import casadi as cs
import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Mass = 0.033  # mass [kg]
g = 9.81  # gravity [m/s^2]
u_eq = np.array([Mass * g, 0, 0])  # equilibrium input [N, rad/s, rad/s]

nx, nu = 10, 3
# Define states.
x = cs.MX.sym('x')
x_dot = cs.MX.sym('x_dot')
y = cs.MX.sym('y')
y_dot = cs.MX.sym('y_dot')
z = cs.MX.sym('z')
z_dot = cs.MX.sym('z_dot')
phi = cs.MX.sym('phi')  # roll angle [rad]
phi_dot = cs.MX.sym('phi_dot')
theta = cs.MX.sym('theta')  # pitch angle [rad]
theta_dot = cs.MX.sym('theta_dot')
X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, phi_dot, theta_dot)
# Define input collective thrust and theta.
T = cs.MX.sym('T_c')  # normalized thrust [N]
R = cs.MX.sym('R_c')  # desired roll angle [rad]
P = cs.MX.sym('P_c')  # desired pitch angle [rad]
U = cs.vertcat(T, R, P)
# The thrust in PWM is converted from the normalized thrust.
# With the formulat F_desired = b_F * T + a_F
params_acc = [20.907574256269616, 3.653687545690674]
params_roll_rate = [-130.3, -16.33, 119.3]
params_pitch_rate = [-99.94, -13.3, 84.73]
psi = 0

a =  20.907574256269616
b =  3.653687545690674
c =  -130.3
d =  -16.33
e =  119.3
f =  -99.94
h =  -13.3
l =  84.73

# Define dynamics equations.
# TODO: create a parameter for the new quad model
X_dot = cs.vertcat(x_dot,
                    (a * T + b) * (
                                cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)),
                    y_dot,
                    (a * T + b) * (
                                cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)),
                    z_dot,
                    (a * T + b) * cs.cos(phi) * cs.cos(theta) - g,
                    phi_dot,
                    theta_dot,
                    c * phi + d * phi_dot + e * R,
                    f * theta + h * theta_dot + l * P,
                        )
# Define observation.
Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, phi_dot, theta_dot)


def get_linearized_model(dt):
    """Linearize the system around the hovering equilibrium point."""
    # Equilibrium point
    X_eq = np.zeros(nx)
    T_eq = Mass * g 
    U_eq = np.array([T_eq, 0, 0])

    # Linearize dynamics using CasADi's jacobian
    A = cs.jacobian(X_dot, X)
    B = cs.jacobian(X_dot, U)

    # Create CasADi functions for Jacobians
    A_fun = cs.Function('A_fun', [X, U], [A])
    B_fun = cs.Function('B_fun', [X, U], [B])

    # Evaluate Jacobians at equilibrium
    A_val = A_fun(X_eq, U_eq)
    B_val = B_fun(X_eq, U_eq)

    # Discretize the system using forward Euler
    Ad = np.eye(nx) + A_val * dt
    Bd = B_val * dt
    
    return Ad, Bd, U_eq


def lqr(A, B, Q, R):
    """Solve the discrete-time LQR controller."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K


def generate_figure_eight_trajectory(t_vec, traj_period=25.0, scaling=1.0, z_height=1.0):
    """Generate a figure-eight trajectory in the xy-plane."""
    traj_freq = 2.0 * np.pi / traj_period
    
    x_ref = scaling * np.sin(traj_freq * t_vec)
    y_ref = scaling * np.sin(traj_freq * t_vec) * np.cos(traj_freq * t_vec)
    z_ref = np.full_like(t_vec, z_height)
    
    x_dot_ref = scaling * traj_freq * np.cos(traj_freq * t_vec)
    y_dot_ref = scaling * traj_freq * (np.cos(traj_freq * t_vec)**2 - np.sin(traj_freq * t_vec)**2)
    z_dot_ref = np.zeros_like(t_vec)
    
    # Reference for angular states is zero
    phi_ref, theta_ref, phi_dot_ref, theta_dot_ref = [np.zeros_like(t_vec) for _ in range(4)]
    
    X_ref = np.vstack([
        x_ref, x_dot_ref, y_ref, y_dot_ref, z_ref, z_dot_ref,
        phi_ref, theta_ref, phi_dot_ref, theta_dot_ref
    ])
    return X_ref


def simulate_quadrotor():
    """Simulate the quadrotor dynamics and track the figure-eight trajectory."""
    # Time settings
    dt = 1/60
    T_sim = 500 # [s]
    t_vec = np.arange(0, T_sim, dt)

    # Get linearized model and equilibrium input
    Ad, Bd, U_eq = get_linearized_model(dt)

    # LQR cost matrices
    Q = np.diag([10, 1, 10, 1, 10, 1, 1, 1, 1, 1])  # Adjusted for better tracking
    # Q = np.diag([50, 1, 50, 1, 50, 1, 1, 1, 1, 1])
    # Q = np.diag([1000, 1, 1000, 1, 1000, 1, 1, 1, 1, 1])  # Adjusted for better tracking
    R = np.diag([0.1, 0.1, 0.1])
    
    # Compute LQR gain
    K = lqr(Ad, Bd, Q, R)

    # Generate reference trajectory
    X_ref = generate_figure_eight_trajectory(t_vec, traj_period=15, scaling=1.0)

    # Initialize simulation variables
    X_sim = np.zeros((nx, len(t_vec)))
    X_sim[:, 0] = X_ref[:, 0]  # Start at the beginning of the trajectory
    U_sim = np.zeros((nu, len(t_vec)))

    # CasADi function for nonlinear dynamics
    f = cs.Function('f', [X, U], [X_dot])

    # Simulation loop
    for i in range(len(t_vec) - 1):
        error = X_sim[:, i] - X_ref[:, i]
        delta_U = -K @ error
        action = U_eq + delta_U
        U_sim[:, i] = action.full().flatten() 
        
        
        # Update state using RK4 integration
        k1 = f(X_sim[:, i], U_sim[:, i])
        k2 = f(X_sim[:, i] + dt/2 * k1, U_sim[:, i])
        k3 = f(X_sim[:, i] + dt/2 * k2, U_sim[:, i])
        k4 = f(X_sim[:, i] + dt * k3, U_sim[:, i])
        X_sim[:, i + 1] = X_sim[:, i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4).full().flatten()

    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    ax1.plot(X_ref[0, :], X_ref[2, :], X_ref[4, :], 'r--', label='Reference')
    ax1.plot(X_sim[0, :], X_sim[2, :], X_sim[4, :], 'b-', label='Simulated')
    ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]'); ax1.set_zlabel('Z [m]')
    ax1.legend(); ax1.set_title('3D Trajectory Tracking')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(X_ref[0, :], X_ref[2, :], 'r--', label='Reference')
    ax2.plot(X_sim[0, :], X_sim[2, :], 'b-', label='Simulated')
    ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
    ax2.legend(); ax2.set_title('XY Plane Trajectory'); ax2.axis('equal')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('quadrotor_trajectory_tracking.png', dpi=300)

    # Plot full state trajectory
    fig_states, axs = plt.subplots(5, 2, figsize=(12, 10), sharex=True)
    state_labels = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot', 'phi', 'theta', 'phi_dot', 'theta_dot']
    units = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s', 'rad', 'rad', 'rad/s', 'rad/s']
    for i, (ax, label, unit) in enumerate(zip(axs.flat, state_labels, units)):
        ax.plot(t_vec, X_ref[i, :], 'r--', label='Reference')
        ax.plot(t_vec, X_sim[i, :], 'b-', label='Simulated')
        ax.set_ylabel(f'{label} [{unit}]')
        ax.legend()
        ax.grid(True)
    fig_states.suptitle('State Trajectories over Time')
    axs.flat[-1].set_xlabel('Time [s]')
    axs.flat[-2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig('quadrotor_states.png', dpi=300)

    # Plot action trajectory
    fig_actions, axs = plt.subplots(nu, 1, figsize=(10, 6), sharex=True)
    if nu == 1:
        axs = [axs]
    action_labels = ['Thrust', 'Desired Roll', 'Desired Pitch']
    action_units = ['N', 'rad', 'rad']
    for i, (ax, label, unit) in enumerate(zip(axs, action_labels, action_units)):
        ax.plot(t_vec[:-1], U_sim[i, :-1], 'b-')
        ax.set_ylabel(f'{label} [{unit}]')
        ax.grid(True)
    axs[-1].set_xlabel('Time [s]')
    fig_actions.suptitle('Action Trajectories over Time')
    plt.tight_layout()
    plt.savefig('quadrotor_actions.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    simulate_quadrotor()