'''Linear Time-Invariant (LTI) Model Predictive Control using Acados.'''
from copy import deepcopy

import casadi as cs
import numpy as np
import scipy
import matplotlib.pyplot as plt
from termcolor import colored

from safe_control_gym.controllers.mpc.mpc_acados import MPC_ACADOS
from safe_control_gym.controllers.mpc.mpc_utils import set_acados_constraint_bound
from safe_control_gym.utils.utils import timing
from safe_control_gym.envs.constraints import BoundedConstraint

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError as e:
    print(colored(f'Error: {e}', 'red'))
    print(colored('acados not installed, cannot use acados-based controller. Exiting.', 'red'))
    print(colored('- To build and install acados, follow the instructions at https://docs.acados.org/installation/index.html', 'yellow'))
    print(colored('- To set up the acados python interface, follow the instructions at https://docs.acados.org/python_interface/index.html', 'yellow'))
    print()
    exit()

class LinearMPC_ACADOS(MPC_ACADOS):
    '''MPC with linear time-invariant (LIT) model.'''

    def __init__(
            self,
            env_func,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            warmstart: bool = True,
            soft_constraints: bool = False,
            soft_penalty: float = 10000,
            terminate_run_on_done: bool = True,
            constraint_tol: float = 1e-6,
            # runner args
            # shared/base args
            output_dir: str = 'results/temp',
            additional_constraints: list = None,
            use_gpu: bool = False,
            seed: int = 0,
            use_RTI: bool = False,
            compute_initial_guess_method = 'lqr',
            use_lqr_gain_and_terminal_cost: bool = False,
            use_r_term: bool = False,
            **kwargs
    ):
        '''Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
            use_RTI (bool): Real-time iteration for acados.
            use_lqr_gain_and_terminal_cost (bool): Use LQR ancillary gain and terminal cost for the MPC.
            use_r_term (bool): Use r term correction for linearization error in MPC dynamics.
        '''
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})
        super().__init__(
            env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=soft_constraints,
            soft_penalty=soft_penalty,
            terminate_run_on_done=terminate_run_on_done,
            constraint_tol=constraint_tol,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            compute_initial_guess_method=compute_initial_guess_method,
            use_lqr_gain_and_terminal_cost=use_lqr_gain_and_terminal_cost,
            use_gpu=use_gpu,
            seed=seed,
            **kwargs
        )

        self.x_guess = None
        self.u_guess = None
        # linearization point
        self.x_lin = np.atleast_2d(self.model.X_EQ)[0, :].T
        self.u_lin = np.atleast_2d(self.model.U_EQ)[0, :].T
        # acados settings
        self.use_RTI = use_RTI

    def setup_acados_model(self) -> AcadosModel:
        '''Sets up symbolic model for acados.'''
        acados_model = super().setup_acados_model()
        
        # Linear dynamics: Δx_{i,k+1} = A_{i,k}Δx_{i,k} + B_{i,k}Δu_{i,k}
        f_linear = self.linear_dynamics_func(acados_model.x, acados_model.u)
        
        if self.use_r_term:
            # Set up parameters for reference trajectory when using r term
            nx, nu = self.model.nx, self.model.nu
            
            # Parameters: [x_ref_k, u_ref_k, x_ref_k+1]
            # We need both current and next reference states to compute r term
            x_ref_k = cs.MX.sym('x_ref_k', nx)
            u_ref_k = cs.MX.sym('u_ref_k', nu)
            x_ref_k_plus_1 = cs.MX.sym('x_ref_k_plus_1', nx)
            
            # Concatenate all parameters
            acados_model.p = cs.vertcat(x_ref_k, u_ref_k, x_ref_k_plus_1)
            
            # Compute r term: r_i,k = f(x_ref_i,k, u_ref_i,k) - x_ref_i,k+1
            # Use the original nonlinear discrete dynamics
            fc_func = self.model.fc_func
            k1 = fc_func(x_ref_k, u_ref_k)
            k2 = fc_func(x_ref_k + self.dt / 2 * k1, u_ref_k)
            k3 = fc_func(x_ref_k + self.dt / 2 * k2, u_ref_k)
            k4 = fc_func(x_ref_k + self.dt * k3, u_ref_k)
            f_nonlinear_ref = x_ref_k + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            
            # Compute r term: difference between nonlinear dynamics and reference trajectory
            r_term = f_nonlinear_ref - x_ref_k_plus_1
            
            # Complete dynamics with r term: Δx_{i,k+1} = A_{i,k}Δx_{i,k} + B_{i,k}Δu_{i,k} + r_{i,k}
            acados_model.disc_dyn_expr = f_linear + r_term
        else:
            # Standard linear dynamics without r term
            acados_model.disc_dyn_expr = f_linear
        
        return acados_model

    def setup_acados_optimizer(self, acados_model: AcadosModel) -> AcadosOcp:
        '''Sets up linearized optimization problem.'''
        # do not use the parent class method for debugging
        # ocp = super().setup_acados_optimizer(acados_model)
        
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = acados_model

        # set dimensions
        ocp.dims.N = self.T  # prediction horizon

        # set cost (NOTE: safe-control-gym uses quadratic cost)
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(self.Q / self.dt, self.R / self.dt)
        ocp.cost.W_e = self.Q if not self.use_lqr_gain_and_terminal_cost else self.P
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:(nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # placeholder y_ref and y_ref_e (will be set in select_action)
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))
        # Constraints are overridden with delta constraints
        # general constraint expressions
        state_constraint_expr_list = []
        input_constraint_expr_list = []
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            state_constraint_expr_list.append(state_constraint(ocp.model.x+self.x_lin))
        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            input_constraint_expr_list.append(input_constraint(ocp.model.u+self.u_lin))

        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(*state_constraint_expr_list)  # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        ocp = self.processing_acados_constraints_expression(ocp, h0_expr, h_expr, he_expr)
        # for state_constraint in self.constraints.state_constraints:
        #     if isinstance(state_constraint, BoundedConstraint):
        #         ocp.constraints.lbx = state_constraint.lower_bounds - self.x_lin
        #         ocp.constraints.ubx = state_constraint.upper_bounds - self.x_lin
        #         ocp.constraints.idxbx = np.arange(nx)
        #         ocp.constraints.lbx_e = state_constraint.lower_bounds - self.x_lin
        #         ocp.constraints.ubx_e = state_constraint.upper_bounds - self.x_lin
        #         ocp.constraints.idxbx_e = np.arange(nx)
        #     else:
        #         raise ValueError('Constraint type not supported. Support only for BoundedConstraint and descendants. Check constraints.py.')
        # for input_constraint in self.constraints.input_constraints:
        #     if isinstance(input_constraint, BoundedConstraint):
        #         ocp.constraints.lbu = input_constraint.lower_bounds - self.u_lin.flatten()
        #         ocp.constraints.ubu = input_constraint.upper_bounds - self.u_lin.flatten()
        #         ocp.constraints.idxbu = np.arange(nu)
        #     else:
        #         raise ValueError('Constraint type not supported. Support only for BoundedConstraint and descendants. Check constraints.py.')

        # slack costs for nonlinear constraints (same treatment as MPC_ACADOS)
        if self.soft_constraints:
            # slack variables for all constraints
            ocp.constraints.Jsh_0 = np.eye(h0_expr.shape[0])
            ocp.constraints.Jsh = np.eye(h_expr.shape[0])
            ocp.constraints.Jsh_e = np.eye(he_expr.shape[0])
            # slack penalty
            L2_pen = self.soft_penalty
            L1_pen = self.soft_penalty
            ocp.cost.Zl_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.zl_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.zu_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.zl = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.zu = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.Zu_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.zl_e = L1_pen * np.ones(he_expr.shape[0])
            ocp.cost.zu_e = L1_pen * np.ones(he_expr.shape[0])

        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # Initialize parameters for reference trajectory only if using r term
        if self.use_r_term:
            # Parameters: [x_ref_k, u_ref_k, x_ref_k+1] for each stage
            n_params = 2 * nx + nu  # x_ref_k (nx) + u_ref_k (nu) + x_ref_k+1 (nx)
            ocp.parameter_values = np.zeros((n_params,))

        # set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' if not self.use_RTI else 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 25 if not self.use_RTI else 1
        # ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH' if not self.use_RTI else 'MERIT_BACKTRACKING'
        # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        ocp.solver_options.tf = self.T * self.dt  # prediction horizon
        ocp.code_export_directory = self.output_dir + '/linear_mpc_c_generated_code'

        return ocp

    # @timing
    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solves linear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        
        '''
        nx, nu = self.model.nx, self.model.nu
        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, 'lbx', obs - self.x_lin)
        self.acados_ocp_solver.set(0, 'ubx', obs - self.x_lin)

        # warm-starting solver
        # NOTE: only for ipopt warm-starting; since acados
        # has a built-in warm-starting mechanism.
        if self.warmstart:
            if self.x_guess is None or self.u_guess is None:
                # compute initial guess with IPOPT
                self.compute_initial_guess(obs)
            for idx in range(self.T + 1):
                init_x = self.x_guess[:, idx] - self.x_lin.flatten()
                self.acados_ocp_solver.set(idx, 'x', init_x)
            for idx in range(self.T):
                if nu == 1:
                    init_u = np.array([self.u_guess[idx]]) - self.u_lin.flatten()
                else:
                    init_u = self.u_guess[:, idx] - self.u_lin.flatten()
                self.acados_ocp_solver.set(idx, 'u', init_u)

        # set reference for the control horizon
        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1

        x_ref = goal_states[:, :-1] - np.repeat(self.x_lin.reshape(-1, 1), self.T, axis=1)
        u_ref = np.repeat(self.U_EQ.reshape(-1, 1) - self.u_lin.reshape(-1, 1), self.T, axis=1)
        y_ref = np.concatenate((x_ref, u_ref), axis=0)
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, 'yref', y_ref[:, idx])
        y_ref_e = goal_states[:, -1] - self.x_lin.flatten()
        self.acados_ocp_solver.set(self.T, 'yref', y_ref_e)

        # Set parameters for r term computation at each stage (only if using r term)
        if self.use_r_term:
            # Parameters: [x_ref_k, u_ref_k, x_ref_k+1] for each stage
            for idx in range(self.T):
                x_ref_k = goal_states[:, idx]  # Current reference state (absolute)
                u_ref_k = self.U_EQ.reshape(-1)  # Current reference input (absolute)
                x_ref_k_plus_1 = goal_states[:, idx + 1]  # Next reference state (absolute)
                
                # Combine parameters: [x_ref_k, u_ref_k, x_ref_k+1]
                p_values = np.concatenate([x_ref_k, u_ref_k, x_ref_k_plus_1])
                self.acados_ocp_solver.set(idx, 'p', p_values)

        # solve the optimization problem
        try:
            if self.use_RTI:
                # preparation phase
                self.acados_ocp_solver.options_set('rti_phase', 1)
                status = self.acados_ocp_solver.solve()

                # feedback phase
                self.acados_ocp_solver.options_set('rti_phase', 2)
                status = self.acados_ocp_solver.solve()
            else:
                status = self.acados_ocp_solver.solve()

            # get the open-loop solution
            if self.x_prev is None and self.u_prev is None:
                self.x_prev = np.zeros((nx, self.T + 1))
                self.u_prev = np.zeros((nu, self.T))
            if self.u_prev is not None and nu == 1:
                self.u_prev = self.u_prev.reshape((1, -1))
            for i in range(self.T + 1):
                self.x_prev[:, i] = self.acados_ocp_solver.get(i, 'x')
            for i in range(self.T):
                self.u_prev[:, i] = self.acados_ocp_solver.get(i, 'u')
            if nu == 1:
                self.u_prev = self.u_prev.flatten()

            # get the solver status
            n_sqp_iter = self.acados_ocp_solver.get_stats('sqp_iter')
            n_qp_iter = self.acados_ocp_solver.get_stats('qp_iter')
            # print(f'acados returned status {status}. SQP iterations: {n_sqp_iter}. QP iterations: {n_qp_iter}.')

        except Exception:
            print(colored('Infeasible MPC Problem', 'red'))
            # get the solver status
            self.acados_ocp_solver.print_statistics()
            status = self.acados_ocp_solver.get_stats('status')
            print(f'acados returned status {status}. ')
        action = self.acados_ocp_solver.get(0, 'u')

        # Store the solution for warm-starting and results
        self.x_guess = self.x_prev
        self.u_guess = self.u_prev
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['goal_states'].append(deepcopy(goal_states))
        self.results_dict['inference_time'].append(self.acados_ocp_solver.get_stats("time_tot"))

        self.prev_action = action

        # get the open-loop solution
        if self.x_prev is None and self.u_prev is None:
            self.x_prev = np.zeros((nx, self.T + 1))
            self.u_prev = np.zeros((nu, self.T))
        if self.u_prev is not None and nu == 1:
            self.u_prev = self.u_prev.reshape((1, -1))
        for i in range(self.T + 1):
            self.x_prev[:, i] = self.acados_ocp_solver.get(i, 'x')
        for i in range(self.T):
            self.u_prev[:, i] = self.acados_ocp_solver.get(i, 'u')
        if nu == 1:
            self.u_prev = self.u_prev.flatten()

        self.x_guess = self.x_prev
        self.u_guess = self.u_prev
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['goal_states'].append(deepcopy(goal_states))

        # recover the action
        action += self.u_lin.flatten()
        
        # self._compute_open_loop_prediction()
        # self.plot_open_loop_prediction()

        if self.use_lqr_gain_and_terminal_cost:
            action += self.lqr_gain @ (obs - self.x_prev[:, 0])

        return action


    # def _compute_open_loop_prediction(self):
    #     """
    #     Compute the open-loop prediction from the current MPC solution.
    #     Stores the prediction in self.open_loop_states and self.open_loop_inputs.
    #     """
    #     if self.x_prev is None or self.u_prev is None:
    #         return
        
    #     # Convert delta states/inputs back to absolute states/inputs
    #     self.open_loop_states = self.x_prev + np.repeat(self.x_lin.reshape(-1, 1), self.T + 1, axis=1)
        
    #     if self.model.nu == 1:
    #         # Handle 1D input case
    #         u_prev_reshaped = self.u_prev.reshape(1, -1) if self.u_prev.ndim == 1 else self.u_prev
    #         self.open_loop_inputs = u_prev_reshaped + np.repeat(self.u_lin.reshape(-1, 1), self.T, axis=1)
    #     else:
    #         self.open_loop_inputs = self.u_prev + np.repeat(self.u_lin.reshape(-1, 1), self.T, axis=1)

    # def plot_open_loop_prediction(self, fig=None, show_states=True, show_inputs=True, 
    #                              state_labels=None, input_labels=None):
    #     """
    #     Plot the open-loop prediction from the MPC solution.
        
    #     Args:
    #         fig: matplotlib figure to plot on (creates new if None)
    #         show_states (bool): whether to plot predicted states
    #         show_inputs (bool): whether to plot predicted inputs  
    #         state_labels (list): labels for state variables
    #         input_labels (list): labels for input variables
            
    #     Returns:
    #         fig: matplotlib figure object
    #     """
        
    #     if not hasattr(self, 'open_loop_states') or self.open_loop_states is None:
    #         print("No open-loop prediction available yet. Run select_action first.")
    #         return None
        
    #     if fig is None:
    #         fig = plt.figure(figsize=(12, 8))
        
    #     nx, nu = self.model.nx, self.model.nu
    #     time_horizon = np.arange(self.T + 1)
    #     input_time_horizon = np.arange(self.T)
        
    #     # Use environment state labels if none provided
    #     if state_labels is None and hasattr(self.env, 'STATE_LABELS'):
    #         state_labels = self.env.STATE_LABELS
        
    #     # Use environment action labels if none provided  
    #     if input_labels is None and hasattr(self.env, 'ACTION_LABELS'):
    #         input_labels = self.env.ACTION_LABELS
        
    #     # Determine subplot layout
    #     n_plots = 0
    #     if show_states and nx > 0:
    #         n_plots += 1
    #     if show_inputs and nu > 0:
    #         n_plots += 1
            
    #     if n_plots == 0:
    #         return fig
        
    #     plot_idx = 1
        
    #     # Plot states
    #     if show_states and nx > 0:
    #         ax_states = fig.add_subplot(n_plots, 1, plot_idx)
    #         for i in range(nx):
    #             label = state_labels[i] if state_labels and i < len(state_labels) else f'State {i+1}'
    #             ax_states.plot(time_horizon, self.open_loop_states[i, :], 'o-', label=label)
    #         ax_states.set_xlabel('Time Step')
    #         ax_states.set_ylabel('State Value')
    #         ax_states.set_title('Open-Loop State Prediction')
    #         ax_states.legend()
    #         ax_states.grid(True)
    #         plot_idx += 1
        
    #     # Plot inputs
    #     if show_inputs and nu > 0:
    #         ax_inputs = fig.add_subplot(n_plots, 1, plot_idx)
    #         for i in range(nu):
    #             label = input_labels[i] if input_labels and i < len(input_labels) else f'Input {i+1}'
    #             if nu == 1:
    #                 input_data = self.open_loop_inputs if self.open_loop_inputs.ndim == 1 else self.open_loop_inputs[i, :]
    #             else:
    #                 input_data = self.open_loop_inputs[i, :]
    #             ax_inputs.step(input_time_horizon, input_data, where='post', label=label)
    #         ax_inputs.set_xlabel('Time Step')
    #         ax_inputs.set_ylabel('Input Value')
    #         ax_inputs.set_title('Open-Loop Input Prediction')
    #         ax_inputs.legend()
    #         ax_inputs.grid(True)
        
    #     plt.tight_layout()
    #     # plt.show()
    #     plt.savefig('./linear_mpc_open_loop_prediction.png')
    #     plt.close()
    #     print("Saved open-loop prediction figure to './linear_mpc_open_loop_prediction.png'")
