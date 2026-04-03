from copy import deepcopy
import time

import casadi as cs
import numpy as np

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import (
    ConstraintList,
    GENERAL_CONSTRAINTS,
    create_constraint_list,
)


def euler_discrete(f, n, m, k, dt):
    """Euler discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        k (int): parameter dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    """
    X = cs.SX.sym("X", n)
    U = cs.SX.sym("U", m)
    P = cs.SX.sym("P", k)
    x_next = X + dt * f(X, U, P)
    eu_dyn = cs.Function("eu_f", [X, U, P], [x_next], ["x0", "u", "p"], ["xf"])

    return eu_dyn


def rk_discrete(f, n, m, k, dt):
    """Runge Kutta discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        k (int): parameter dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    """
    X = cs.SX.sym("X", n)
    U = cs.SX.sym("U", m)
    P = cs.SX.sym("P", k)
    # Runge-Kutta 4 integration
    k1 = f(X, U, P)
    k2 = f(X + dt / 2 * k1, U, P)
    k3 = f(X + dt / 2 * k2, U, P)
    k4 = f(X + dt * k3, U, P)
    x_next = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    rk_dyn = cs.Function("rk_f", [X, U, P], [x_next], ["x0", "u", "p"], ["xf"])

    return rk_dyn


def _create_semi_definite_matrix(n):
    # U = cs.SX.sym("U", cs.Sparsity.lower(n))
    # u = cs.vertcat(*U.nonzeros())
    # W_upper = cs.Function("Lower_tri_W", [u], [U])
    # np = int(n * (n + 1) / 2)
    # p = cs.MX.sym("p", np)
    # W = W_upper(p)
    # WW = W.T @ W

    n_param = n
    P = cs.MX.sym("P", n)
    W = cs.diag(P)
    # WW = cs.sqrt(W.T @ W)
    return W, P, n_param


def update_initial_guess(x_prev, u_prev, sigma_prev, opt_vars_fn):
    # shift previous solutions by 1 step
    u_guess = deepcopy(u_prev)
    x_guess = deepcopy(x_prev)
    sigma_guess = deepcopy(sigma_prev)
    u_guess[:, :-1] = u_guess[:, 1:]
    x_guess[:, :-1] = x_guess[:, 1:]
    sigma_guess[:, :-1] = sigma_guess[:, 1:]
    opt_vars_init = opt_vars_fn(x_guess, u_guess, sigma_guess).full()
    return opt_vars_init


def reset_constraints(constraints):
    """Set up the constraints list.

    Args:
        constraints (list): List of constraints the controller is subject to.

    Returns:
        constraints_list (ConstraintList): List of constraints.
        state_constraints_sym (list): Symbolic state constraints.
        input_constraints_sym (list): Symbolic input constraints.
    """

    constraints_list = ConstraintList(constraints)
    state_constraints_sym = constraints_list.get_state_constraint_symbolic_models()
    input_constraints_sym = constraints_list.get_input_constraint_symbolic_models()
    if len(constraints_list.input_state_constraints) > 0:
        raise NotImplementedError(
            "[Error] Cannot handle combined state input constraints yet."
        )
    return constraints_list, state_constraints_sym, input_constraints_sym


class MPCFunction:
    def __init__(
        self,
        env_fun,
        gamma,
        model,
        horizon: int = 5,
        warmstart: bool = True,
        soft_constraints: bool = True,
        constraint_tol: float = 1e-6,
        additional_constraints: list = None,
        n_parallel_solver: int = 1,
        n_train_solver: int = 1,
        jit: bool = False,
        jit_options: dict = None,
    ):
        self.env = env_fun
        self.model = model
        self.dt = self.model.dt
        self.T = horizon
        self.gamma = gamma
        self.update_step_count = 0
        self.soft_constraints = soft_constraints
        self.constraint_tol = constraint_tol
        self.warmstart = warmstart
        self.n_parallel_solver = n_parallel_solver
        self.n_train_solver = n_train_solver
        self.jit = jit
        self.jit_options = jit_options if jit_options is not None else {}

        # Constraint list
        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(
                additional_constraints, GENERAL_CONSTRAINTS, self.env
            )
            self.additional_constraints = additional_ConstraintsList.constraints
            (
                self.constraints,
                self.state_constraints_sym,
                self.input_constraints_sym,
            ) = reset_constraints(
                self.env.constraints.constraints + self.additional_constraints
            )
        else:
            (
                self.constraints,
                self.state_constraints_sym,
                self.input_constraints_sym,
            ) = reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []

        # Additional entries
        self.u_prev = None
        self.x_prev = None
        self.sigma_prev = None
        self.x_goal = None
        self.mode = None
        self.traj = None
        self.traj_step = 0
        self.infos = [None] * self.n_parallel_solver
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = "stabilization"
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = "tracking"
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0

        # Setup optimizer
        self.solver_dict = None
        self.dynamics_func = None
        self.set_dynamics_func()
        self.setup_optimizer()

    def reset(self, idx=None):
        # Previously solved states & inputs, useful for warm start.
        self.u_prev = None
        self.x_prev = None
        self.sigma_prev = None
        self.x_goal = None
        self.traj = None
        self.traj_step = 0
        if idx is not None:
            self.infos[idx] = None
        else:
            self.infos = [None] * self.n_parallel_solver

        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = "stabilization"
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = "tracking"
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0

    def add_constraints(self, constraints):
        """Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        """
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = (
            reset_constraints(constraints + self.constraints.constraints)
        )

    def remove_constraints(self, constraints):
        """Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        """
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, ValueError(
                "This constraint is not in the current list of constraints"
            )
            old_constraints_list.remove(constraint)
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = (
            reset_constraints(old_constraints_list)
        )

    def set_dynamics_func(self):
        """Updates symbolic dynamics with actual control frequency."""
        self.dynamics_func = rk_discrete(
            self.model.param_fc_func,
            self.model.nx,
            self.model.nu,
            self.model.npl,
            self.dt,
        )
        # self.dynamics_func = euler_discrete(
        #     self.model.param_fc_func,
        #     self.model.nx,
        #     self.model.nu,
        #     self.model.npl,
        #     self.dt,
        # )

    def setup_optimizer(self):
        """Sets up nonlinear optimization problem."""
        nx, nu, npl = self.model.nx, self.model.nu, self.model.npl
        T = self.T
        etau = 1e-5  # barrier parameter for interior point method
        start_time = time.time()

        # Optimization variable: [x0, u0, sigma0, x1, u1, ...]
        opt_vars = []
        x_var, u_var, sigma_var = [], [], []
        for i in range(T):
            x = cs.MX.sym("x", nx)
            u = cs.MX.sym("u", nu)
            sigma = cs.MX.sym("sigma", nx)
            # append the casadi var to opt_Vars
            opt_vars.append(x)
            opt_vars.append(u)
            opt_vars.append(sigma)
            # append the casadi var to individual vars for ease of use
            x_var.append(x)
            u_var.append(u)
            sigma_var.append(sigma)
        x = cs.MX.sym("x", nx)  # final state
        sigma = cs.MX.sym("sigma", nx)
        opt_vars.append(x)
        opt_vars.append(sigma)
        x_var.append(x)
        sigma_var.append(sigma)
        # compile the variable vectors/matrices
        opt_vars = cs.vcat(opt_vars)
        x_var, u_var, sigma_var = cs.hcat(x_var), cs.hcat(u_var), cs.hcat(sigma_var)
        # function definitions for conversion
        opt_vars_fn = cs.Function("opt_vars_fun", [x_var, u_var, sigma_var], [opt_vars])
        xus_fn = cs.Function("xus_fun", [opt_vars], [x_var, u_var, sigma_var])
        opt_act_fn = cs.Function("opt_act_fun", [opt_vars], [u_var[:, 0]])

        # Parameters
        # Fixed parameters
        # Initial state.
        x_init = cs.MX.sym("x_init", nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = cs.MX.sym("x_ref", nx, T + 1)
        fixed_param = cs.vertcat(x_init)
        ref_param = cs.reshape(x_ref, -1, 1)

        # Learnable parameters
        # Cost
        Qk, th_q, _ = _create_semi_definite_matrix(nx)
        Rk, th_r, _ = _create_semi_definite_matrix(nu)
        Qt, th_qt, _ = _create_semi_definite_matrix(nx)
        # theta_param = cs.MX.sym("theta_var", nq + nr)
        cost_param = cs.vertcat(th_q, th_r, th_qt)
        back_off_param = cs.MX.sym("back_off_param", nx)
        # Model
        model_param = cs.MX.sym("f_param", npl)

        # cost (cumulative)
        cost = 0
        w = 1e3 * np.ones((1, nx))
        cost_func = self.model.loss
        for i in range(T):
            cost += (
                self.gamma**i
                * cost_func(
                    x=x_var[:, i],
                    u=u_var[:, i],
                    Xr=x_ref[:, i],
                    Ur=np.zeros((nu, 1)),
                    Q=Qk,
                    R=Rk,
                )["l"]
            )
        # Terminal cost.
        cost += (
            self.gamma**T
            * cost_func(
                x=x_var[:, -1],
                u=np.zeros((nu, 1)),
                Xr=x_ref[:, -1],
                Ur=np.zeros((nu, 1)),
                Q=Qt,
                R=np.zeros((nu, nu)),
            )["l"]
        )
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
        lm = cs.MX.sym("lm", nx)
        mult.append(lm)
        lamb.append(lm)
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(
                x0=x_var[:, i], u=u_var[:, i], p=model_param
            )["xf"]
            con_list.append(x_var[:, i + 1] - next_state)
            con_lbg.append(cs.DM.zeros(nx, 1))
            con_ubg.append(cs.DM.zeros(nx, 1))
            con_eq += [True] * nx

            H_eq.append(x_var[:, i + 1] - next_state)
            lm = cs.MX.sym("lm", nx)
            mult.append(lm)
            lamb.append(lm)

            # State bounds
            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                cost += w @ sigma_var[:, i]
                con_list.append(
                    state_constraint(x_var[:, i])[:nx]
                    - sigma_var[:, i]
                    + back_off_param
                )
                con_list.append(
                    state_constraint(x_var[:, i])[nx:]
                    - sigma_var[:, i]
                    + back_off_param
                )
                con_list.append(-sigma_var[:, i])
                con_lbg.append(-cs.DM.inf(3 * nx, 1))
                con_ubg.append(cs.DM.zeros(3 * nx, 1))
                con_eq += [False] * 3 * nx

                H_ieq.append(
                    state_constraint(x_var[:, i])[:nx]
                    - sigma_var[:, i]
                    + back_off_param
                )
                H_ieq.append(
                    state_constraint(x_var[:, i])[nx:]
                    - sigma_var[:, i]
                    + back_off_param
                )
                H_ieq.append(-sigma_var[:, i])
                lm = cs.MX.sym("lm", 3 * nx)
                mult.append(lm)
                mu.append(lm)

            # Action bounds
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                con_list.append(input_constraint(u_var[:, i]) + self.constraint_tol)
                con_lbg.append(-cs.DM.inf(2 * nu, 1))
                con_ubg.append(cs.DM.zeros(2 * nu, 1))
                con_eq += [False] * 2 * nu

                H_ieq.append(input_constraint(u_var[:, i]) + self.constraint_tol)
                lm = cs.MX.sym("lm", 2 * nu)
                mult.append(lm)
                mu.append(lm)

        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            cost += w @ sigma_var[:, -1]
            con_list.append(
                state_constraint(x_var[:, -1])[:nx] - sigma_var[:, -1] + back_off_param
            )
            con_list.append(
                state_constraint(x_var[:, -1])[nx:] - sigma_var[:, -1] + back_off_param
            )
            con_list.append(-sigma_var[:, -1])
            con_lbg.append(-cs.DM.inf(3 * nx, 1))
            con_ubg.append(cs.DM.zeros(3 * nx, 1))
            con_eq += [False] * 3 * nx

            H_ieq.append(
                state_constraint(x_var[:, -1])[:nx] - sigma_var[:, -1] + back_off_param
            )
            H_ieq.append(
                state_constraint(x_var[:, -1])[nx:] - sigma_var[:, -1] + back_off_param
            )
            H_ieq.append(-sigma_var[:, -1])
            lm = cs.MX.sym("lm", 3 * nx)
            mult.append(lm)
            mu.append(lm)
        # concatenating all the lists
        con_list, H_eq, H_ieq = cs.vcat(con_list), cs.vcat(H_eq), cs.vcat(H_ieq)
        mult, lamb, mu = cs.vcat(mult), cs.vcat(lamb), cs.vcat(mu)
        con_lbg, con_ubg = cs.vcat(con_lbg), cs.vcat(con_ubg)
        lang_mult_fn = cs.Function("lang_mult_fn", [mult], [lamb, mu])

        # Create solver (IPOPT solver in this version)
        opts_setting = {
            "print_time": 0,
            "record_time": True,
            "expand": True,
            "equality": con_eq,
            "structure_detection": "auto",
            "debug": False,
            "fatrop.mu_init": etau,
            "fatrop.max_iter": 500,
            "fatrop.print_level": 0,
            "fatrop.acceptable_tol": 1e-5,
        }
        jit_opts = {
            "jit": self.jit,
            "jit_cleanup": False,
            "jit_temp_suffix": False,
            "jit_options": self.jit_options,
        }
        opts_setting.update(jit_opts)
        vnlp_prob = {
            "f": cost,
            "x": opt_vars,
            "p": cs.vertcat(
                fixed_param, ref_param, cost_param, back_off_param, model_param
            ),
            "g": con_list,
        }
        pisolver = cs.nlpsol("pisolver", "fatrop", vnlp_prob, opts_setting)
        print(
            f"[MPC Setup] NLP problem setup time: {time.time() - start_time:.3f} seconds."
        )
        start_time = time.time()

        # Build Lagrangian
        lagrangian = cost + cs.transpose(lamb) @ H_eq + cs.transpose(mu) @ H_ieq
        dlag_dw = cs.jacobian(lagrangian, opt_vars)
        # Build KKT matrix
        R_kkt = cs.vertcat(
            cs.transpose(dlag_dw),
            H_eq,
            mu * H_ieq + etau,
        )
        # z contains all variables of the lagrangian
        z = cs.vertcat(opt_vars, mult)
        theta = cs.vertcat(cost_param, back_off_param, model_param)

        #### Sensitivities for value function
        lagrangian_fn = cs.Function(
            "Lagrangian", [z, fixed_param, ref_param, theta], [lagrangian]
        )
        dlag_fn = lagrangian_fn.factory(
            "dV", ["i0", "i1", "i2", "i3"], ["jac:o0:i2", "jac:o0:i3"]
        )
        [dVdref, dVdtheta] = dlag_fn(z, fixed_param, ref_param, theta)
        # Sensitivity against theta
        dVdtheta_fn = cs.Function(
            "dVdtheta_fn", [z, fixed_param, ref_param, theta], [dVdtheta.T]
        )

        #### Generate sensitivity of the KKT matrix
        rkkt_fn = cs.Function("rkkt_fn", [z, fixed_param, ref_param, theta], [R_kkt])
        rkkt_norm_fn = cs.Function(
            "rkkt_norm_fn",
            [z, fixed_param, ref_param, theta],
            [cs.norm_2(R_kkt)],
            jit_opts,
        )
        dR_sensfunc = rkkt_fn.factory(
            "dR", ["i0", "i1", "i2", "i3"], ["jac:o0:i0", "jac:o0:i2", "jac:o0:i3"]
        )
        [dRdz, dRdP_ref, dRdP_theta] = dR_sensfunc(z, fixed_param, ref_param, theta)
        # dRdP = cs.horzcat(dRdP_ref, dRdP_theta)
        dRdP = cs.horzcat(dRdP_theta)  # only learnable param
        print(
            f"[MPC Setup] KKT matrix setup time: {time.time() - start_time:.3f} seconds."
        )
        start_time = time.time()

        # Generate sensitivity of the optimal solution
        # 1. Pseudo inverse method (less efficient when there are many decision variables)
        # dzdP = -cs.inv(dRdz) @ dRdP
        # 2. Linear solver method (more efficient, especially for large problems, since it can exploit sparsity)
        dzdP = -cs.solve(dRdz, dRdP)
        # dPi = dzdP[nx: nx + nu, :].T
        dPi = dzdP
        # 3. Adjoint method (most efficient when there are many parameters, but requires additional implementation effort)
        # S = cs.DM.zeros(nu, dRdz.shape[0])
        # for i in range(nu):
        #     S[i, nx + i] = 1.0
        # dPi_prime = cs.solve(dRdz.T, S.T).T
        # dPi = -(dPi_prime @ dRdP).T
        dPi_zeros = cs.MX.zeros(dPi.shape)
        f_true = cs.Function("f_true", [z, fixed_param, ref_param, theta], [dPi])
        f_false = cs.Function(
            "f_false", [z, fixed_param, ref_param, theta], [dPi_zeros]
        )
        dPi_fn = cs.Function.if_else("dPi_fn", f_true, f_false)
        # dPi_fn.save('dPi_fn.casadi')
        print(
            f"[MPC Setup] Sensitivity setup time: {time.time() - start_time:.3f} seconds."
        )
        start_time = time.time()

        # R_kkt function with jit
        all_fn = cs.Function(
            "all_fn",
            [z, fixed_param, ref_param, theta],
            [cs.norm_2(R_kkt), dPi],
            jit_opts,
        )
        # all_fn.save("all_fn.casadi")
        print(
            f"[MPC Setup] JIT compilation time: {time.time() - start_time:.3f} seconds."
        )

        self.solver_dict = {
            "x_var": x_var,
            "u_var": u_var,
            "state_slack": sigma_var,
            "opt_vars": opt_vars,
            "mult": mult,
            "fixed_param": fixed_param,
            "ref_param": ref_param,
            "theta_param": theta,
            "opt_vars_fn": opt_vars_fn,
            "xus_fn": xus_fn,
            "opt_act_fn": opt_act_fn,
            "cost": cost,
            "lower_bound": con_lbg,
            "upper_bound": con_ubg,
            "lang_mult_fn": lang_mult_fn,
            "solver": pisolver,
            "R_kkt": R_kkt,
            "dVdtheta": dVdtheta,
            "dVdtheta_fn": dVdtheta_fn,
            "rkkt_fn": rkkt_fn,
            "rkkt_norm_fn": rkkt_norm_fn,
            "dpi_fn": dPi_fn,
            "all_fn": all_fn,
            "jit_options": jit_opts,
        }

    def get_references(self, traj_step=None, traj_ref=None):
        """Constructs reference states along mpc horizon.(nx, T+1)."""
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            if traj_step is None:
                traj_step = self.traj_step
            if traj_ref is None:
                traj_ref = self.traj
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(traj_step, self.traj.shape[-1])
            end = min(traj_step + self.T + 1, self.traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            goal_states = np.concatenate(
                [traj_ref[:, start:end], np.tile(traj_ref[:, -1:], (1, remain))], -1
            )
        else:
            raise Exception("Reference for this mode is not implemented.")
        return goal_states  # (nx, T+1).

    def select_action(self, obs, theta, traj_ref, info=None, mode="eval"):
        """Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            theta (ndarray): Learnable param based on current state
            traj_ref (ndarray): Learnable trajectory
            info (dict): Current info
            mode (string): Current mode of evaluation (eval vs train)

        Returns:
            action (ndarray): Input/action to the task/env.
        """
        solver_dict = self.solver_dict
        solver = solver_dict["solver"]
        opt_vars_fn = solver_dict["opt_vars_fn"]
        xus_fn = solver_dict["xus_fn"]

        # Collect the fixed param
        # Assign reference trajectory within horizon.
        fixed_param = obs[: self.model.nx, None]
        # goal_states = self.get_references(self.traj_step, traj_ref)
        goal_states = traj_ref[0].copy()
        ref_param = goal_states.T.reshape(-1, 1)
        # Collect learnable parameters
        p_param = np.concatenate((fixed_param, ref_param, theta[:, None]))[:, 0]
        if self.mode == "tracking":
            self.traj_step += 1

        opt_vars_init = np.zeros(
            (solver_dict["opt_vars"].shape[0], solver_dict["opt_vars"].shape[1])
        )
        if self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            opt_vars_init = update_initial_guess(
                self.x_prev, self.u_prev, self.sigma_prev, opt_vars_fn
            )

        # Solve the optimization problem.
        soln = solver(
            x0=opt_vars_init,
            p=p_param,
            lbg=solver_dict["lower_bound"],
            ubg=solver_dict["upper_bound"],
        )
        optimal = solver.stats()["success"]

        # Post-processing the solution
        opt_vars = soln["x"].full()
        x_val, u_val, sigma_val = xus_fn(opt_vars)
        self.x_prev = x_val.full()
        self.u_prev = u_val.full()
        self.sigma_prev = sigma_val.full()
        results_dict = {
            "horizon_states": deepcopy(self.x_prev),
            "horizon_inputs": deepcopy(self.u_prev),
            "goal_states": deepcopy(ref_param),
            "t_wall": solver.stats()["t_wall_total"],
        }

        # Take the first action from the solved action sequence.
        if self.u_prev.ndim > 1:
            action = self.u_prev[:, 0]
        else:
            action = np.array([self.u_prev[0]])

        # additional info
        info = {
            "success": optimal,
            "soln": deepcopy(soln),
            "fixed_param": deepcopy(fixed_param),
            "ref_param": deepcopy(ref_param),
            "theta_param": deepcopy(theta),
            "traj_step": deepcopy(self.traj_step) - 1,
        }
        return action, info, results_dict, optimal
