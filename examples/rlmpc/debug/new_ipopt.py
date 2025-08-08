import numpy as np
import casadi as cs

nx, nu, npl = 4, 2, 1
T = 20
etau = 1e-4
gamma = 0.99
lb_x, ub_x = np.array([-2.0, -2.0, -5.0, -5.0]), np.array([2.0, 2.0, 5.0, 5.0])
lb_u, ub_u = np.array([-1.0, -1.0]), np.array([1.0, 1.0])


def _create_semi_definite_matrix(n):
    # np = n
    P = cs.MX.sym('P', n)
    W = cs.diag(P)
    WW = cs.sqrt(W.T @ W)
    return WW, P, n


def cost_func(x, u, q, r):
    return x.T @ q @ x + u.T @ r @ u


def dynamics_func(x, u, p):
    A = cs.DM([[1.0, 0.0, 0.1, 0.0], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    B = cs.DM([[0.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    x_next = A @ x + p[0] * B @ u
    return x_next


# # Define optimizer and variables.
# # States.
# x_var = cs.MX.sym('x_var', nx, T + 1)
# # Inputs.
# u_var = cs.MX.sym('u_var', nu, T)
# # Add slack variables
# state_slack = cs.MX.sym('sigma_var', nx, T + 1)
# opt_vars = cs.vertcat(cs.reshape(u_var, -1, 1),
#                       cs.reshape(x_var, -1, 1),
#                       cs.reshape(state_slack, -1, 1))
# opt_vars_fn = cs.Function('opt_vars_fun', [opt_vars], [x_var, u_var, state_slack])

# Handling optimization variable
opt_vars = []
x_var, u_var, sigma_var = [], [], []
for i in range(T):
    x = cs.MX.sym('x', nx)
    u = cs.MX.sym('u', nu)
    sigma = cs.MX.sym('sigma', nx)
    opt_vars.append(x)
    opt_vars.append(u)
    opt_vars.append(sigma)
    x_var.append(x)
    u_var.append(u)
    sigma_var.append(sigma)
x = cs.MX.sym('x', nx)  # final state
sigma = cs.MX.sym('sigma', nx)
opt_vars.append(x)
opt_vars.append(sigma)
x_var.append(x)
sigma_var.append(sigma)
opt_vars = cs.vcat(opt_vars)
x_var, u_var, sigma_var = cs.hcat(x_var), cs.hcat(u_var), cs.hcat(sigma_var)
opt_vars_fn = cs.Function('opt_vars_fun', [opt_vars], [x_var, u_var, sigma_var])
opt_act_fn = cs.Function('opt_act_fn', [opt_vars], [u_var[:, 0]])

# Parameters
# Fixed parameters
# Initial state.
x_init = cs.MX.sym('x_init', nx, 1)
# Reference (equilibrium point or trajectory, last step for terminal cost).
# x_ref = cs.MX.sym('x_ref', nx, T + 1)
fixed_param = cs.vertcat(x_init)
# ref_param = cs.reshape(x_ref, -1, 1)

# Learnable parameters
# Cost
Q, th_q, nq = _create_semi_definite_matrix(nx)
R, th_r, nr = _create_semi_definite_matrix(nu)
Qt, th_qt, nqt = _create_semi_definite_matrix(nx)
# theta_param = cs.MX.sym("theta_var", nq + nr)
cost_param = cs.vertcat(th_q, th_r, th_qt)
# Model
model_param = cs.MX.sym('f_param', npl)

# cost (cumulative)
cost = 0
w = 1e3 * np.ones((1, nx))
# cost_func = self.model.loss
for i in range(T):
    # Can ignore the first state cost since fist x_var == x_init.
    cost += gamma ** i * cost_func(x_var[:, i], u_var[:, i], Q, R)
# Terminal cost.
cost += gamma ** T * cost_func(x_var[:, -1], np.zeros(nu), Qt, np.zeros((nu, nu)))
# # Constraints
# g, hu, hx, hs = [], [], [], []
# # initial condition constraints
# g.append(x_var[:, 0] - x_init)
# for i in range(T):
#     # Dynamics constraints.
#     next_state = dynamics_func(x_var[:, i], u_var[:, i], model_param)
#     g.append(x_var[:, i + 1] - next_state)
#     cost += w @ state_slack[:, i]
#     hx.append(- x_var[:, i] - lb_x - state_slack[:, i])
#     hx.append(x_var[:, i] - ub_x - state_slack[:, i])
#     hs.append(- state_slack[:, i])
#     hu.append(- u_var[:, i] - lb_u)
#     hu.append(u_var[:, i] - ub_u)
# # Final state constraints.
# cost += w @ state_slack[:, -1]
# hx.append(- x_var[:, -1] - lb_x - state_slack[:, -1])
# hx.append(x_var[:, -1] - ub_x - state_slack[:, -1])
# hs.append(- state_slack[:, -1])
# # Setting casadi constraints and bounds
# G = cs.vertcat(*g)
# Hu = cs.vertcat(*hu)
# Hx = cs.vertcat(*hx)
# Hs = cs.vertcat(*hs)
# constraint_exp = cs.vertcat(*g, *hu, *hx, *hs)
# lbg = [0] * G.shape[0] + [-np.inf] * (Hu.shape[0] + Hx.shape[0] + Hs.shape[0])
# ubg = [0] * G.shape[0] + [0] * (Hu.shape[0] + Hx.shape[0] + Hs.shape[0])
# lbg = cs.vertcat(*lbg)
# ubg = cs.vertcat(*ubg)

# Constraints
con_list, con_lbg, con_ubg, con_eq = [], [], [], []
H_eq, H_ieq = [], []
mult, lamb, mu = [], [], []
# initial condition constraints
con_list.append(x_var[:, 0] - x_init)
con_lbg.append(cs.DM.zeros(nx, 1))
con_ubg.append(cs.DM.zeros(nx, 1))
con_eq += [True] * nx

H_eq.append(x_var[:, 0] - x_init)
lm = cs.MX.sym('lm', nx)
mult.append(lm)
lamb.append(lm)
for i in range(T):
    # Dynamics constraints.
    next_state = dynamics_func(x_var[:, i], u_var[:, i], model_param)
    con_list.append(x_var[:, i + 1] - next_state)
    con_lbg.append(cs.DM.zeros(nx, 1))
    con_ubg.append(cs.DM.zeros(nx, 1))
    con_eq += [True] * nx

    H_eq.append(x_var[:, i + 1] - next_state)
    lm = cs.MX.sym('lm', nx)
    mult.append(lm)
    lamb.append(lm)

    # State bounds
    cost += w @ sigma_var[:, i]
    con_list.append(-x_var[:, i] + lb_x - sigma_var[:, i])
    con_list.append(x_var[:, i] - ub_x - sigma_var[:, i])
    con_list.append(-sigma_var[:, i])
    con_lbg.append(-cs.DM.inf(3 * nx, 1))
    con_ubg.append(cs.DM.zeros(3 * nx, 1))
    con_eq += [False] * 3 * nx

    H_ieq.append(-x_var[:, i] + lb_x - sigma_var[:, i])
    H_ieq.append(x_var[:, i] - ub_x - sigma_var[:, i])
    H_ieq.append(-sigma_var[:, i])
    lm = cs.MX.sym('lm', 3 * nx)
    mult.append(lm)
    mu.append(lm)

    # Action bounds
    con_list.append(- u_var[:, i] + lb_u)
    con_list.append(u_var[:, i] - ub_u)
    con_lbg.append(-cs.DM.inf(2 * nu, 1))
    con_ubg.append(cs.DM.zeros(2 * nu, 1))
    con_eq += [False] * 2 * nu

    H_ieq.append(- u_var[:, i] + lb_u)
    H_ieq.append(u_var[:, i] - ub_u)
    lm = cs.MX.sym('lm', 2 * nu)
    mult.append(lm)
    mu.append(lm)

# Final state constraints.
cost += w @ sigma_var[:, -1]
con_list.append(-x_var[:, -1] + lb_x - sigma_var[:, -1])
con_list.append(x_var[:, -1] - ub_x - sigma_var[:, -1])
con_list.append(-sigma_var[:, -1])
con_lbg.append(-cs.DM.inf(3 * nx, 1))
con_ubg.append(cs.DM.zeros(3 * nx, 1))
con_eq += [False] * 3 * nx

H_ieq.append(- x_var[:, -1] + lb_x - sigma_var[:, -1])
H_ieq.append(x_var[:, -1] - ub_x - sigma_var[:, -1])
H_ieq.append(-sigma_var[:, -1])
lm = cs.MX.sym('lm', 3 * nx)
mult.append(lm)
mu.append(lm)
# casadi vcats
con_list, H_eq, H_ieq = cs.vcat(con_list), cs.vcat(H_eq), cs.vcat(H_ieq)
mult, lamb, mu = cs.vcat(mult), cs.vcat(lamb), cs.vcat(mu)
con_lbg, con_ubg = cs.vcat(con_lbg), cs.vcat(con_ubg)
lang_mult_fn = cs.Function('lang_mult_fn', [mult], [lamb, mu])

# Create solver (IPOPT solver in this version)
opts_setting = {
    'ipopt.max_iter': 200,
    'ipopt.print_level': 0,
    'print_time': 0,
    'record_time': True,
    'ipopt.mu_target': etau,
    'ipopt.mu_init': etau,
    'ipopt.acceptable_tol': 1e-4,
    'ipopt.acceptable_obj_change_tol': 1e-4,
}
# opts_setting = {
#     'print_time': 0,
#     'record_time': True,
#     'expand': True,
#     'equality': con_eq,
#     'structure_detection': 'auto',
#     'debug': False,
#     # 'jit': True,
#     # 'jit_temp_suffix': False,
#     # 'jit_options.flags': ['-03'],
#     # 'jit_options.compiler': 'ccache gcc',
#     'fatrop.max_iter': 200,
#     'fatrop.print_level': 0,
#     'fatrop.acceptable_tol': 1e-4,
# }
vnlp_prob = {
    'f': cost,
    'x': opt_vars,
    'p': cs.vertcat(fixed_param, cost_param, model_param),
    'g': con_list,
}
vsolver = cs.nlpsol('vsolver', 'ipopt', vnlp_prob, opts_setting)

lagrangian = (
        cost
        + cs.transpose(lamb) @ H_eq
        + cs.transpose(mu) @ H_ieq
)
dlag_dw = cs.jacobian(lagrangian, opt_vars)
dlag_dp1 = cs.jacobian(lagrangian, cost_param)
dlag_dp2 = cs.jacobian(lagrangian, model_param)

# z contains all variables of the lagrangian
z = cs.vertcat(opt_vars, lamb, mu)

dl1 = cs.Function('dldw', [z, fixed_param, cost_param, model_param], [dlag_dw])
dl2 = cs.Function('dldp1', [z, fixed_param, cost_param, model_param], [dlag_dp1])
dl3 = cs.Function('dldp2', [z, fixed_param, cost_param, model_param], [dlag_dp2])

# Build KKT matrix
R_kkt = cs.vertcat(
    cs.transpose(dlag_dw),
    H_eq,
    mu * H_ieq + etau,
)

# Generate sensitivity of the KKT matrix #fixed_param, ref_param, cost_param, model_param
rkkt_fn = cs.Function('rkkt_fn', [z, fixed_param, cost_param, model_param], [R_kkt])
dR_sensfunc = rkkt_fn.factory(
    'dR', ['i0', 'i1', 'i2', 'i3'], ['jac:o0:i0', 'jac:o0:i2', 'jac:o0:i3']
)
[dRdz, dRdP_cost, dRdP_model] = dR_sensfunc(z, fixed_param, cost_param, model_param)
dRdP = cs.horzcat(dRdP_cost, dRdP_model)

dr1 = cs.Function('dRdz', [z, fixed_param, cost_param, model_param], [dRdz])
dr2 = cs.Function('dRdz', [z, fixed_param, cost_param, model_param], [dRdP])

# Generate sensitivity of the optimal solution
dzdP = -cs.inv(dRdz) @ dRdP
dPi = cs.Function('dPi', [z, fixed_param, cost_param, model_param], [dzdP[nx:nx+nu, :].T])


# Test
x0 = np.zeros(2*nx*(T+1) + nu*T)
x = np.array([1.0, 1.0, -0.2, 0.2])
q = np.array([10.0, 10.0, 1.0, 1.0])
r = np.array([1.0, 1.0])
qt = np.array([10.0, 10.0, 1.0, 1.0])
p = np.array([1.0])
param = np.concatenate((x, q, r, qt, p))
soln = vsolver(x0=x0, p=param, lbg=con_lbg, ubg=con_ubg)
stats = vsolver.stats()
print(stats['success'])
print(soln['f'])

opt_vars2 = soln['x'].full()
x_val, u_val, sigma_val = opt_vars_fn(opt_vars2)
print(x_val)
print(u_val)
print(sigma_val)
print(u_val[:, 0])

opt_vars2 = soln['x'].full()
mult = soln['lam_g'].full()
lamb, mu = lang_mult_fn(mult)
z = np.concatenate((opt_vars2, lamb, mu), axis=0)
cost_param = np.concatenate((q, r, qt))

dldw = dl1(z, x, cost_param, p).full()
dldp1 = dl2(z, x, cost_param, p).full()
dldp2 = dl3(z, x, cost_param, p).full()
print("test sensitivities")
print(np.linalg.norm(dldw))
print(dldp1)
print(dldp2)

dr11 = dr1(z, x, cost_param, p).full()
dr22 = dr2(z, x, cost_param, p).full()
# print(dr11)
# print(dr22)

nabla_pi = dPi(z, x, cost_param, p).full()
# nabla_pi_ref = nabla_pi[:, :ref_param.shape[0]]
# nabla_pi_cost = nabla_pi[:, ref_param.shape[0]:ref_param.shape[0] + cost_param.shape[0]]
nabla_pi_model = nabla_pi[cost_param.shape[0]:, :]
print(nabla_pi)

