'''Linear Time-Invariant (LTI) Model Predictive Control using Acados.'''
from copy import deepcopy

import casadi as cs
import numpy as np
from termcolor import colored

from safe_control_gym.controllers.mpc.mpc_acados import MPC_ACADOS
from safe_control_gym.controllers.mpc.mpc_utils import set_acados_constraint_bound
from safe_control_gym.utils.utils import timing

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
        # override the dynamics function with linearized dynamics
        f_disc = self.linear_dynamics_func(acados_model.x, 
                                           acados_model.u)
        acados_model.disc_dyn_expr = f_disc
        return acados_model

    def setup_acados_optimizer(self, acados_model: AcadosModel) -> AcadosOcp:
        '''Sets up linearized optimization problem.'''
        ocp = super().setup_acados_optimizer(acados_model)
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
        self.acados_ocp_solver.set(0, 'lbx', obs)
        self.acados_ocp_solver.set(0, 'ubx', obs)

        # warm-starting solver
        # NOTE: only for ipopt warm-starting; since acados
        # has a built-in warm-starting mechanism.
        if self.warmstart:
            if self.x_guess is None or self.u_guess is None:
                # compute initial guess with IPOPT
                self.compute_initial_guess(obs)
            for idx in range(self.T + 1):
                init_x = self.x_guess[:, idx] - self.x_lin
                self.acados_ocp_solver.set(idx, 'x', init_x)
            for idx in range(self.T):
                if nu == 1:
                    init_u = np.array([self.u_guess[idx]])
                else:
                    init_u = self.u_guess[:, idx] - self.u_lin
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
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, 'yref', y_ref_e)

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

        if self.use_lqr_gain_and_terminal_cost:
            action += self.lqr_gain @ (obs - self.x_prev[:, 0])

        return action
