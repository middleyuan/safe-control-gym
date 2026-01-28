'''SAC utilities.'''

from collections import defaultdict, deque
from copy import deepcopy
import time

import casadi as cs
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.controllers.mpc.mpc_utils import reset_constraints
from safe_control_gym.controllers.rlmpc.rlmpc_utils import AdamOptimizer, rk_discrete
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list
from safe_control_gym.math_and_models.distributions import Normal
from safe_control_gym.math_and_models.neural_networks import MLP


class SAC_MPC_Agent:
    '''A SAC class that encapsulates models, optimizers and update functions.'''

    def __init__(self,
                 env_fun,
                 obs_space,
                 act_space,
                 gamma,
                 model,
                 hidden_dim=256,
                 tau=0.005,
                 init_temperature=0.2,
                 use_entropy_tuning=False,
                 target_entropy=None,
                 exploration_init=-2.0,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 entropy_lr=0.001,
                 activation='relu',
                 actor_config=None,
                 update_freq=1,
                 **kwargs):

        # Parameters.
        self.env = env_fun
        self.obs_space = obs_space
        self.act_space = act_space
        self.exploration_init = exploration_init
        self.gamma = gamma
        self.tau = tau
        self.use_entropy_tuning = use_entropy_tuning
        self.activation = activation

        # Model.
        self.ac = MLPActorCritic(
            self.env,
            obs_space,
            act_space,
            gamma,
            model,
            hidden_dims=[hidden_dim] * 2,
            exploration_init=self.exploration_init,
            activation=self.activation,
            actor_config=actor_config,
        )
        self.log_alpha = torch.tensor(np.log(init_temperature))

        if self.use_entropy_tuning:
            self.log_alpha.requires_grad = True
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(act_space.shape).item()
            else:
                self.target_entropy = target_entropy

        # target networks
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Optimizers.
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.ac.q1.parameters()) + list(self.ac.q2.parameters()), critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=entropy_lr)
        self.update_freq = update_freq
        self.count = 0

    @property
    def alpha(self):
        '''Entropy-tuning parameter/temperature'''
        return self.log_alpha.exp()

    def to(self, device):
        '''Puts agent to device.'''
        self.ac.to(device)
        self.ac_targ.to(device)
        self.log_alpha.to(device)

    def train(self):
        '''Sets training mode.'''
        self.ac.train()

    def eval(self):
        '''Sets evaluation mode.'''
        self.ac.eval()

    def reset(self, idx=None):
        '''Reset function, especially needed for resetting MPC actor'''
        self.ac.reset(idx)

    def state_dict(self):
        '''Snapshots agent state.'''
        return {
            'ac': self.ac.state_dict(),
            'log_alpha': self.log_alpha,
            'ac_targ': self.ac_targ.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict()
        }

    def load_state_dict(self, state_dict, strict=True):
        '''Restores agent state.'''
        self.ac.load_state_dict(state_dict['ac'], strict=strict)
        self.log_alpha = state_dict['log_alpha']
        self.ac_targ.load_state_dict(state_dict['ac_targ'], strict=strict)
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])
        self.alpha_opt.load_state_dict(state_dict['alpha_opt'])

    def compute_policy_loss(self, batch, batch_th):
        '''Returns policy loss(es) given batch of data.'''
        obs_th = batch_th['obs']
        obs, info = batch['obs'], batch['info']
        act_th, logp, action_th, nabla_pi_ref, nabla_pi_theta, optimal = self.ac.actor.forward_train(obs, info)

        '''Returns policy loss(es) given batch of data.'''
        q1 = self.ac.q1(obs_th, act_th)
        q2 = self.ac.q2(obs_th, act_th)
        q = torch.min(q1, q2)
        policy_loss = (self.alpha.detach() * logp - q)
        policy_loss = torch.where(optimal > 0.9, policy_loss, torch.nan).nanmean()

        entropy_loss = torch.zeros(1)
        if self.use_entropy_tuning:
            entropy_loss = -(self.log_alpha * (logp + self.target_entropy).detach())
            entropy_loss = torch.where(optimal > 0.9, entropy_loss, torch.nan).nanmean()
        return policy_loss, entropy_loss, action_th, nabla_pi_ref, nabla_pi_theta, optimal

    def compute_q_loss(self, batch, batch_th):
        '''Returns q-value loss(es) given batch of data.'''
        obs, act, next_obs = batch_th['obs'], batch_th['act'], batch_th['next_obs']
        rew, mask = batch_th['rew'], batch_th['mask']
        next_obs_np = np.array(batch['next_obs'])
        info = batch['info']
        q1 = self.ac.q1(obs, act)
        q2 = self.ac.q2(obs, act)

        with torch.no_grad():
            next_act, next_logp, _, _, _, optimal = self.ac.actor.forward_train(
                next_obs_np, info, update_info=True, compute_sensitivities=False,
            )
            next_q1_targ = self.ac_targ.q1(next_obs, next_act)
            next_q2_targ = self.ac_targ.q2(next_obs, next_act)
            next_q_targ = torch.min(next_q1_targ, next_q2_targ)
            # q value regression target
            q_targ = rew + self.gamma * mask * (next_q_targ - self.alpha * next_logp)

        q1_loss = (q1 - q_targ).pow(2)
        q2_loss = (q2 - q_targ).pow(2)
        critic_loss = (q1_loss + q2_loss)
        critic_loss = torch.where(optimal > 0.9, critic_loss, torch.nan).nanmean()
        return critic_loss

    def update(self, batch, batch_th, device='cpu'):
        '''Updates model parameters based on current training batch.'''
        results = defaultdict(list)

        # actor update
        if self.count % self.update_freq == 0:
            (policy_loss, entropy_loss, action_th, nabla_pi_ref, nabla_pi_theta, optimal) \
                = self.compute_policy_loss(batch, batch_th)
            self.actor_opt.zero_grad()
            policy_loss.backward()

            # Passing the gradients through the mpc
            theta = self.ac.actor.get_theta_param(batch_th['obs'])
            theta_loss = (action_th.grad.unsqueeze(1) @ nabla_pi_theta @ theta.unsqueeze(2)).sum()
            # traj_ref = self.ac.actor.get_ref_param(batch['info'])
            # ref_loss = action_th.grad.unsqueeze(1) @ nabla_pi_ref @ traj_ref.unsqueeze(2)
            theta_loss.backward()
            self.actor_opt.step()
            with torch.no_grad():
                self.ac.actor.mpc_param.clamp_(1e-5, 100.0)
                # self.ac.actor.logstd.clamp_(-3.5, -0.5)

            if self.use_entropy_tuning:
                self.alpha_opt.zero_grad()
                entropy_loss.backward()
                self.alpha_opt.step()

            results['policy_loss'] = policy_loss.item()
            results['entropy_loss'] = entropy_loss.item()
            results['alpha'] = self.alpha.item()
            results['theta_loss'] = theta_loss.item()
            # results['exploration_std'] = self.ac.actor.logstd.exp().mean().item()      

        # critic update
        critic_loss = self.compute_q_loss(batch, batch_th)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        results['critic_loss'] = critic_loss.item()

        # update target networks
        if self.count % self.update_freq == 0:
            soft_update(self.ac, self.ac_targ, self.tau)
        self.count += 1
        return results


# -----------------------------------------------------------------------------------
#                   Models
# -----------------------------------------------------------------------------------


class MLPActorCritic(nn.Module):
    '''Model for the actor-critic agent.

    Attributes:
        actor (MLPActor): policy network.
        critic (MLPCritic): value network.
    '''

    def __init__(self,
                 env,
                 obs_space,
                 act_space,
                 gamma,
                 model,
                 hidden_dims=(64, 64),
                 exploration_init=-1.0,
                 activation='tanh',
                 actor_config=None
                 ):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            raise Exception('PPO-MPC is currently only implemented for continuous action spaces')
        # Policy.
        self.actor = MPCActor(
            env, obs_dim, act_dim, act_space, hidden_dims, activation, gamma, model, exploration_init, actor_config
        )
        # Q functions
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_dims, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_dims, activation)

    def step(self, obs, info=None):
        a, soln_info = self.actor(obs, actor_info=info)
        return a.cpu().numpy(), soln_info

    def act(self, obs, info=None):
        a, _ = self.actor(obs, deterministic=True, actor_info=info)
        return a.cpu().numpy()

    def reset(self, idx=None):
        self.actor.reset(idx)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dims, activation):
        super().__init__()
        self.q_net = MLP(obs_dim + act_dim, 1, hidden_dims, activation)

    def forward(self, obs, act):
        return self.q_net(torch.cat([obs, act], dim=-1))


class MPCActor(nn.Module):
    '''Actor MPC model.'''

    def __init__(self, env, obs_dim, act_dim, action_space, hidden_dims, activation, gamma, model, exploration_init, actor_config):
        super().__init__()
        # mpc actor
        self.mpc = MPCPolicyFunction(env, gamma, model, **actor_config['mpc_config'])

        # Parameters
        self.q_mpc = actor_config['q_mpc']
        self.r_mpc = actor_config['r_mpc']
        self.qt_mpc = actor_config['qt_mpc']
        self.back_off = actor_config['back_off']
        self.model_param = actor_config['model_param']
        self._init_param_val()
        self.n_learnable_param = 0
        for k in self.param_dict.keys():
            self.n_learnable_param += self.param_dict[k].shape[0]
        temp = np.concatenate((self.q_mpc, self.r_mpc, self.qt_mpc, self.back_off, self.model_param))
        self.mpc_param = nn.Parameter(torch.FloatTensor(temp))
        self.param_net = MLP(obs_dim, self.n_learnable_param, hidden_dims, activation)
        # self.traj_param = nn.Parameter(torch.FloatTensor(self.mpc.traj))
        self.traj_param = torch.FloatTensor(self.mpc.traj)

        # Construct output action distribution.
        self.net = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation)
        self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)
        # self.log_std = nn.Parameter(exploration_init * torch.ones(act_dim))
        # self.dist_fn = lambda x: Normal(x, self.logstd.exp())
        self.dist_fn = lambda mu, log_std: Normal(mu, log_std.exp())
        self.log_std_min = -20
        self.log_std_max = 2
        self.tanh_squash = True

        # action rescaling (from cleanrl)
        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32).flatten()
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32).flatten()

    def forward(self, obs, deterministic=False, with_logprob=True, actor_info=None):
        theta = self.get_theta_param(obs)
        traj_param = self.get_references(actor_info)
        if obs.ndim > 1:
            act, info, results_dict, optimal_flag = self.mpc.select_action_batch(
                obs, theta.numpy(), traj_param, actor_info
            )
        else:
            act, info, results_dict, optimal_flag = self.mpc.select_action(
                obs, theta.numpy(), traj_param
            )
        act = torch.FloatTensor(np.array(act))
        # optimal_flag = torch.FloatTensor(np.array(optimal_flag))
        if self.tanh_squash:
            act = self.inverse_squashing(act)

        # action distribution
        net_out = self.net(torch.FloatTensor(obs))
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        dist = self.dist_fn(act, log_std)
        if deterministic:
            x_t = dist.mode()
        else:
            x_t = dist.rsample()
        if self.tanh_squash:
            # Squash the output
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
        else:
            action = x_t
        # action = x_t
        return action, info

    def forward_train(self, obs, info, update_info=False, compute_sensitivities=True):
        theta = self.get_theta_param(obs)
        action, nabla_pi_ref, nabla_pi_theta, optimal_flag = self.mpc.select_action_batch_train(
            obs, theta.detach().numpy(), info, update_info=update_info, compute_sensitivities=compute_sensitivities,
        )
        action = torch.FloatTensor(action)
        nabla_pi_ref = torch.FloatTensor(np.array(nabla_pi_ref))
        nabla_pi_theta = torch.FloatTensor(np.array(nabla_pi_theta))
        optimal_flag = torch.FloatTensor(optimal_flag).T if len(optimal_flag) > 0 else torch.FloatTensor(
            np.ones((obs.shape[0], 1))
        )
        action.requires_grad_()
        if self.tanh_squash:
            # Inverse squashing
            z_t = self.inverse_squashing(action)

        # action distribution
        net_out = self.net(torch.FloatTensor(obs))
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        dist = self.dist_fn(action, log_std)
        x_t = dist.rsample()
        logp = dist.log_prob(x_t)

        # Squash the output
        if self.tanh_squash:
            y_t = torch.tanh(x_t)
            act = y_t * self.action_scale + self.action_bias
            logp -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6).sum(-1, keepdim=True)
        else:
            act = x_t
        return act, logp, action, nabla_pi_ref, nabla_pi_theta, optimal_flag

    def _init_param_val(self):
        self.param_dict = {'l': np.concatenate((self.q_mpc, self.r_mpc, self.qt_mpc)),
                           'b': np.array(self.back_off),
                           'f': np.array(self.model_param)}

    def inverse_squashing(self, y):
        """Inverse of tanh squashing function."""
        y = (y - self.action_bias) / self.action_scale
        return torch.atanh(torch.clamp(y, -1.0+1e-6, 1.0-1e-6))

    def reset(self, idx=None):
        self.mpc.reset(idx)

    def get_theta_param(self, obs):
        if obs.ndim > 1:
            theta = self.mpc_param.repeat(obs.shape[0], 1) + 0.0 * self.param_net.forward(torch.FloatTensor(obs))
        else:
            theta = self.mpc_param + 0.0 * self.param_net.forward(torch.FloatTensor(obs))
        theta += torch.rand_like(theta) * 1e-6
        return theta

    def get_references(self, info_batch):
        """Constructs reference states along mpc horizon.(nx, T+1)."""
        goal_states_batch = []
        for info in info_batch:
            traj_step = info['current_step']
            traj_ref = info['x_ref'].T
            if self.mpc.env.TASK == Task.STABILIZATION:
                # Repeat goal state for horizon steps.
                goal_states = np.tile(self.mpc.env.X_GOAL.reshape(-1, 1), (1, self.mpc.T + 1))
            elif self.mpc.env.TASK == Task.TRAJ_TRACKING:
                # Slice trajectory for horizon steps, if not long enough, repeat last state.
                start = min(traj_step, traj_ref.shape[-1])
                end = min(traj_step + self.mpc.T + 1, traj_ref.shape[-1])
                remain = max(0, self.mpc.T + 1 - (end - start))
                goal_states = np.concatenate([
                    traj_ref[:, start:end],
                    np.tile(traj_ref[:, -1:], (1, remain))
                ], -1)
            else:
                raise Exception('Reference for this mode is not implemented.')
            goal_states_batch.append(goal_states)
        return goal_states_batch  # list of (nx, T+1).

    def get_ref_param(self, info_batch):
        goal_states_batch = torch.FloatTensor()
        if self.mpc.env.TASK == Task.TRAJ_TRACKING:
            for info in info_batch:
                traj_step = info['traj_step']
                # Slice trajectory for horizon steps, if not long enough, repeat last state.
                start = min(traj_step, self.mpc.traj.shape[-1])
                end = min(traj_step + self.mpc.T + 1, self.mpc.traj.shape[-1])
                remain = max(0, self.mpc.T + 1 - (end - start))
                goal_states = torch.cat((
                    self.traj_param[:, start:end],
                    torch.tile(self.traj_param[:, -1:], (1, remain))
                ), -1).T.reshape(-1, 1).T
                goal_states_batch = torch.cat((goal_states_batch, goal_states), 0)
        else:
            raise Exception('Reference update for this mode is not implemented.')
        return goal_states_batch  # (nx, T+1).


class MPCPolicyFunction:
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
            additional_ConstraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            (self.constraints, self.state_constraints_sym,
             self.input_constraints_sym) = reset_constraints(self.env.constraints.constraints
                                                             + self.additional_constraints)
        else:
            (self.constraints, self.state_constraints_sym,
             self.input_constraints_sym) = reset_constraints(self.env.constraints.constraints)
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
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0

        # Setup optimizer
        self.solver_dict = None
        self.dynamics_func = None
        self.set_dynamics_func()
        self.setup_optimizer()

        # Parallel solvers
        self.pi_solvers, self.rkkt_norm_fns, _, _ = self.get_parallel_solver(self.n_parallel_solver)
        self.pi_solvers_train, self.rkkt_norm_fns_train, _, self.all_solvers_train = self.get_parallel_solver(self.n_train_solver)

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
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0

    def add_constraints(self, constraints):
        """Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        """
        (self.constraints, self.state_constraints_sym,
         self.input_constraints_sym) = reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self, constraints):
        """Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        """
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, \
                ValueError('This constraint is not in the current list of constraints')
            old_constraints_list.remove(constraint)
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(
            old_constraints_list)

    def set_dynamics_func(self):
        """Updates symbolic dynamics with actual control frequency."""
        self.dynamics_func = rk_discrete(self.model.param_fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.model.npl,
                                         self.dt)
        # self.dynamics_func = euler_discrete(self.model.param_fc_func,
        #                                     self.model.nx,
        #                                     self.model.nu,
        #                                     self.model.npl,
        #                                     self.dt)

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
            x = cs.MX.sym('x', nx)
            u = cs.MX.sym('u', nu)
            sigma = cs.MX.sym('sigma', nx)
            # append the casadi var to opt_Vars
            opt_vars.append(x)
            opt_vars.append(u)
            opt_vars.append(sigma)
            # append the casadi var to individual vars for ease of use
            x_var.append(x)
            u_var.append(u)
            sigma_var.append(sigma)
        x = cs.MX.sym('x', nx)  # final state
        sigma = cs.MX.sym('sigma', nx)
        opt_vars.append(x)
        opt_vars.append(sigma)
        x_var.append(x)
        sigma_var.append(sigma)
        # compile the variable vectors/matrices
        opt_vars = cs.vcat(opt_vars)
        x_var, u_var, sigma_var = cs.hcat(x_var), cs.hcat(u_var), cs.hcat(sigma_var)
        # function definitions for conversion
        opt_vars_fn = cs.Function('opt_vars_fun', [x_var, u_var, sigma_var], [opt_vars])
        xus_fn = cs.Function('xus_fun', [opt_vars], [x_var, u_var, sigma_var])
        opt_act_fn = cs.Function('opt_act_fun', [opt_vars], [u_var[:, 0]])

        # Parameters
        # Fixed parameters
        # Initial state.
        x_init = cs.MX.sym('x_init', nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = cs.MX.sym('x_ref', nx, T + 1)
        fixed_param = cs.vertcat(x_init)
        ref_param = cs.reshape(x_ref, -1, 1)

        # Learnable parameters
        # Cost
        Q, th_q, nq = _create_semi_definite_matrix(nx)
        R, th_r, nr = _create_semi_definite_matrix(nu)
        Qt, th_qt, nqt = _create_semi_definite_matrix(nx)
        # theta_param = cs.MX.sym("theta_var", nq + nr)
        cost_param = cs.vertcat(th_q, th_r, th_qt)
        back_off_param = cs.MX.sym('back_off_param', nx)
        # Model
        model_param = cs.MX.sym('f_param', npl)

        # cost (cumulative)
        cost = 0
        w = 1e3 * np.ones((1, nx))
        cost_func = self.model.loss
        for i in range(T):
            cost += self.gamma ** i * cost_func(x=x_var[:, i],
                                                u=u_var[:, i],
                                                Xr=x_ref[:, i],
                                                Ur=np.zeros((nu, 1)),
                                                Q=Q,
                                                R=R)['l']
        # Terminal cost.
        cost += self.gamma ** T * cost_func(x=x_var[:, -1],
                                            u=np.zeros((nu, 1)),
                                            Xr=x_ref[:, -1],
                                            Ur=np.zeros((nu, 1)),
                                            Q=Qt,
                                            R=np.zeros((nu, nu)))['l']
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
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], u=u_var[:, i], p=model_param)['xf']
            con_list.append(x_var[:, i + 1] - next_state)
            con_lbg.append(cs.DM.zeros(nx, 1))
            con_ubg.append(cs.DM.zeros(nx, 1))
            con_eq += [True] * nx

            H_eq.append(x_var[:, i + 1] - next_state)
            lm = cs.MX.sym('lm', nx)
            mult.append(lm)
            lamb.append(lm)

            # State bounds
            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                cost += w @ sigma_var[:, i]
                con_list.append(state_constraint(x_var[:, i])[:nx] - sigma_var[:, i] + back_off_param)
                con_list.append(state_constraint(x_var[:, i])[nx:] - sigma_var[:, i] + back_off_param)
                con_list.append(-sigma_var[:, i])
                con_lbg.append(-cs.DM.inf(3 * nx, 1))
                con_ubg.append(cs.DM.zeros(3 * nx, 1))
                con_eq += [False] * 3 * nx

                H_ieq.append(state_constraint(x_var[:, i])[:nx] - sigma_var[:, i] + back_off_param)
                H_ieq.append(state_constraint(x_var[:, i])[nx:] - sigma_var[:, i] + back_off_param)
                H_ieq.append(-sigma_var[:, i])
                lm = cs.MX.sym('lm', 3 * nx)
                mult.append(lm)
                mu.append(lm)

            # Action bounds
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                con_list.append(input_constraint(u_var[:, i]) + self.constraint_tol)
                con_lbg.append(-cs.DM.inf(2 * nu, 1))
                con_ubg.append(cs.DM.zeros(2 * nu, 1))
                con_eq += [False] * 2 * nu

                H_ieq.append(input_constraint(u_var[:, i]) + self.constraint_tol)
                lm = cs.MX.sym('lm', 2 * nu)
                mult.append(lm)
                mu.append(lm)

        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            cost += w @ sigma_var[:, -1]
            con_list.append(state_constraint(x_var[:, -1])[:nx] - sigma_var[:, -1] + back_off_param)
            con_list.append(state_constraint(x_var[:, -1])[nx:] - sigma_var[:, -1] + back_off_param)
            con_list.append(-sigma_var[:, -1])
            con_lbg.append(-cs.DM.inf(3 * nx, 1))
            con_ubg.append(cs.DM.zeros(3 * nx, 1))
            con_eq += [False] * 3 * nx

            H_ieq.append(state_constraint(x_var[:, -1])[:nx] - sigma_var[:, -1] + back_off_param)
            H_ieq.append(state_constraint(x_var[:, -1])[nx:] - sigma_var[:, -1] + back_off_param)
            H_ieq.append(-sigma_var[:, -1])
            lm = cs.MX.sym('lm', 3 * nx)
            mult.append(lm)
            mu.append(lm)
        # concatenating all the lists
        con_list, H_eq, H_ieq = cs.vcat(con_list), cs.vcat(H_eq), cs.vcat(H_ieq)
        mult, lamb, mu = cs.vcat(mult), cs.vcat(lamb), cs.vcat(mu)
        con_lbg, con_ubg = cs.vcat(con_lbg), cs.vcat(con_ubg)
        lang_mult_fn = cs.Function('lang_mult_fn', [mult], [lamb, mu])

        # Create solver (IPOPT solver in this version)
        opts_setting = {
            'print_time': 0,
            'record_time': True,
            'expand': True,
            'equality': con_eq,
            'structure_detection': 'auto',
            'debug': False,
            "jit": self.jit,
            "jit_cleanup": False,
            "jit_temp_suffix": False,
            "jit_options": self.jit_options,
            'fatrop.mu_init': etau,
            'fatrop.max_iter': 500,
            'fatrop.print_level': 0,
            'fatrop.acceptable_tol': 1e-5,
        }
        vnlp_prob = {
            'f': cost,
            'x': opt_vars,
            'p': cs.vertcat(fixed_param, ref_param, cost_param, back_off_param, model_param),
            'g': con_list,
        }
        pisolver = cs.nlpsol('pisolver', 'fatrop', vnlp_prob, opts_setting)
        print(f'[MPC Setup] NLP problem setup time: {time.time() - start_time:.3f} seconds.')
        start_time = time.time()

        # Build Lagrangian
        lagrangian = (
            cost
            + cs.transpose(lamb) @ H_eq
            + cs.transpose(mu) @ H_ieq
        )
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

        # Generate sensitivity of the KKT matrix
        rkkt_fn = cs.Function('rkkt_fn', [z, fixed_param, ref_param, theta], [R_kkt])
        rkkt_norm_fn = cs.Function('rkkt_norm_fn', [z, fixed_param, ref_param, theta], [cs.norm_2(R_kkt)])
        dR_sensfunc = rkkt_fn.factory('dR', ['i0', 'i1', 'i2', 'i3'], ['jac:o0:i0', 'jac:o0:i2', 'jac:o0:i3'])
        [dRdz, dRdP_ref, dRdP_theta] = dR_sensfunc(z, fixed_param, ref_param, theta)
        # dRdP = cs.horzcat(dRdP_ref, dRdP_theta)
        dRdP = cs.horzcat(dRdP_theta)  # only learnable param
        print(f'[MPC Setup] KKT matrix setup time: {time.time() - start_time:.3f} seconds.')
        start_time = time.time()

        # Generate sensitivity of the optimal solution
        # dzdP = -cs.inv(dRdz) @ dRdP
        # dzdP = -cs.solve(dRdz, dRdP)
        # dPi = dzdP[nx: nx + nu, :].T
        S = cs.DM.zeros(nu, dRdz.shape[0])
        for i in range(nu):
            S[i, nx + i] = 1.0
        dPi_prime = cs.solve(dRdz.T, S.T).T
        dPi = -(dPi_prime @ dRdP).T
        dPi_zeros = cs.MX.zeros(dPi.shape)
        f_true = cs.Function('f_true', [z, fixed_param, ref_param, theta], [dPi])
        f_false = cs.Function('f_false', [z, fixed_param, ref_param, theta], [dPi_zeros])
        dPi_fn = cs.Function.if_else('dPi_fn', f_true, f_false)
        print(f'[MPC Setup] Sensitivity setup time: {time.time() - start_time:.3f} seconds.')
        start_time = time.time()

        # R_kkt function with jit
        jit_opts = {
            "jit": self.jit,
            "jit_cleanup": False,
            "jit_temp_suffix": False,
            "jit_options": self.jit_options
        }
        all_fn = cs.Function("all_fn", [z, fixed_param, ref_param, theta], [cs.norm_2(R_kkt), dPi], jit_opts)
        # all_fn.save("all_fn.casadi")
        print(f'[MPC Setup] JIT compilation time: {time.time() - start_time:.3f} seconds.')

        self.solver_dict = {
            'x_var': x_var,
            'u_var': u_var,
            'state_slack': sigma_var,
            'opt_vars': opt_vars,
            'opt_vars_fn': opt_vars_fn,
            'xus_fn': xus_fn,
            'opt_act_fn': opt_act_fn,
            'cost': cost,
            'lower_bound': con_lbg,
            'upper_bound': con_ubg,
            'lang_mult_fn': lang_mult_fn,
            'solver': pisolver,
            'rkkt_fn': rkkt_fn,
            'rkkt_norm_fn': rkkt_norm_fn,
            'dpi_fn': dPi_fn,
            'all_fn': all_fn,
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
            start = min(traj_step, traj_ref.shape[-1])
            end = min(traj_step + self.T + 1, traj_ref.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            goal_states = np.concatenate([
                traj_ref[:, start:end],
                np.tile(traj_ref[:, -1:], (1, remain))
            ], -1)
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).

    def select_action(self, obs, theta, traj_ref, info=None, mode='eval'):
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
        solver = solver_dict['solver']
        opt_vars_fn = solver_dict['opt_vars_fn']
        xus_fn = solver_dict['xus_fn']

        # Collect the fixed param
        # Assign reference trajectory within horizon.
        fixed_param = obs[:self.model.nx, None]
        # goal_states = self.get_references(self.traj_step, traj_ref)
        goal_states = traj_ref[0].copy()
        ref_param = goal_states.T.reshape(-1, 1)
        # Collect learnable parameters
        p_param = np.concatenate((fixed_param, ref_param, theta[:, None]))[:, 0]
        if self.mode == 'tracking':
            self.traj_step += 1

        opt_vars_init = np.zeros((solver_dict['opt_vars'].shape[0], solver_dict['opt_vars'].shape[1]))
        if self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            opt_vars_init = update_initial_guess(self.x_prev, self.u_prev, self.sigma_prev, opt_vars_fn)

        # Solve the optimization problem.
        soln = solver(
            x0=opt_vars_init,
            p=p_param,
            lbg=solver_dict['lower_bound'],
            ubg=solver_dict['upper_bound'],
        )
        optimal = solver.stats()['success']

        # Post-processing the solution
        opt_vars = soln['x'].full()
        x_val, u_val, sigma_val = xus_fn(opt_vars)
        self.x_prev = x_val.full()
        self.u_prev = u_val.full()
        self.sigma_prev = sigma_val.full()
        results_dict = {'horizon_states': deepcopy(self.x_prev), 'horizon_inputs': deepcopy(self.u_prev),
                        'goal_states': deepcopy(ref_param), 't_wall': solver.stats()['t_wall_total']}

        # Take the first action from the solved action sequence.
        if self.u_prev.ndim > 1:
            action = self.u_prev[:, 0]
        else:
            action = np.array([self.u_prev[0]])

        # additional info
        info = {
            'success': optimal,
            'soln': deepcopy(soln),
            'fixed_param': deepcopy(fixed_param),
            'ref_param': deepcopy(ref_param),
            'theta_param': deepcopy(theta),
            'traj_step': deepcopy(self.traj_step) - 1
        }
        return action, info, results_dict, optimal

    def select_action_batch(self, obs_batch, theta, traj_ref, agent_info):
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        con_lbg = self.solver_dict['lower_bound']
        con_ubg = self.solver_dict['upper_bound']
        opt_vars_fn = self.solver_dict['opt_vars_fn']
        xus_fn = self.solver_dict['xus_fn']
        traj_step = self.traj_step
        # goal_states = self.get_references(traj_step, traj_ref)
        if self.mode == 'tracking':
            self.traj_step += 1

        # eval_data_batch = []
        x0, fixed_p, ref_p = [], [], []
        # ref_param = goal_states.T.reshape(-1, 1).repeat(obs_batch.shape[0], 1)
        lbg = con_lbg.full().repeat(obs_batch.shape[0], 1)
        ubg = con_ubg.full().repeat(obs_batch.shape[0], 1)
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        for i, obs in enumerate(obs_batch):
            fixed_param = obs[:self.model.nx]
            ref_param = traj_ref[i].T.reshape(-1, 1)[:, 0]
            opt_vars_init = np.zeros(self.solver_dict['opt_vars'].shape)
            if self.infos[i] is not None:  # shift previous solutions by 1 step based on last soln
                opt_vars_init = self.infos[i]['opt_var']
                x_prev, u_prev, sigma_prev = xus_fn(opt_vars_init)
                x_prev, u_prev, sigma_prev = x_prev.full(), u_prev.full(), sigma_prev.full()
                opt_vars_init = update_initial_guess(x_prev, u_prev, sigma_prev, opt_vars_fn)
                if agent_info[i]['current_step'] == 0:
                    opt_vars_init = np.zeros_like(opt_vars_init)

            x0.append(opt_vars_init[:, 0])
            fixed_p.append(fixed_param)
            ref_p.append(ref_param)
        x0, fixed_p, ref_p = np.array(x0).T, np.array(fixed_p).T, np.array(ref_p).T
        p = np.concatenate((fixed_p, ref_p, theta.T), axis=0)

        # Forward pass through solver
        soln_batch = self.pi_solvers(x0=x0, p=p, lbg=lbg, ubg=ubg)
        z = cs.vertcat(soln_batch['x'], soln_batch['lam_g'])
        rkkt_norm_batch = self.rkkt_norm_fns(z, fixed_p, ref_p, theta.T)
        optimal_batch = rkkt_norm_batch.full() < 1e-3

        # Post-processing the solution
        action_batch, results_dict_batch, info_batch = [], [], []
        for i, obs in enumerate(obs_batch):
            opt_vars = soln_batch['x'].full()[:, i]
            x_val, u_val, sigma_val = xus_fn(opt_vars)
            x_prev = x_val.full()
            u_prev = u_val.full()
            sigma_prev = sigma_val.full()
            results_dict = {
                'horizon_states': deepcopy(x_prev),
                'horizon_inputs': deepcopy(u_prev),
                'horizon_slacks': deepcopy(sigma_prev),
                'goal_states': deepcopy(ref_p[:, i])
            }
            # results_dict['t_wall'].append(opti.stats()['t_wall_total'])

            # Take the first action from the solved action sequence.
            if u_prev.ndim > 1:
                action = u_prev[:, 0]
            else:
                action = np.array([u_prev[0]])

            # additional info
            info = {
                'success': optimal_batch[0, i],
                'opt_var': opt_vars,
                'fixed_param': deepcopy(fixed_p[:, i]),
                'ref_param': deepcopy(ref_p[:, i]),
                'theta_param': deepcopy(theta[i, :]),
                'traj_step': deepcopy(agent_info[i]['current_step']),
                'x_ref': deepcopy(agent_info[i]['x_ref']),
            }

            # result batch
            action_batch.append(action)
            results_dict_batch.append(results_dict)
            info_batch.append(info)
        self.infos = deepcopy(info_batch)
        return action_batch, info_batch, results_dict_batch, optimal_batch

    def select_action_batch_train(self, obs_batch, theta, info_batch, update_info=False, compute_sensitivities=True):
        con_lbg = self.solver_dict['lower_bound']
        con_ubg = self.solver_dict['upper_bound']
        opt_act_fn = self.solver_dict['opt_act_fn']
        opt_vars_fn = self.solver_dict['opt_vars_fn']
        xus_fn = self.solver_dict['xus_fn']

        x0, fixed_p, ref_p = [], [], []
        lbg = con_lbg.full().repeat(obs_batch.shape[0], 1)
        ubg = con_ubg.full().repeat(obs_batch.shape[0], 1)
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        for i, obs in enumerate(obs_batch):
            info = info_batch[i]
            opt_vars_init = info['opt_var']
            fixed_param = obs[:self.model.nx]
            ref_param = info['ref_param']
            if update_info:
                # update the optimization variable init
                x_prev, u_prev, sigma_prev = xus_fn(opt_vars_init)
                x_prev, u_prev, sigma_prev = x_prev.full(), u_prev.full(), sigma_prev.full()
                opt_vars_init = update_initial_guess(x_prev, u_prev, sigma_prev, opt_vars_fn)[:, 0]
                # update the reference parameter
                ref_param = self.get_references(info['traj_step'] + 1, info['x_ref'].T).T.reshape(-1, 1)[:, 0]

            x0.append(opt_vars_init)
            fixed_p.append(fixed_param)
            ref_p.append(ref_param)
        x0, fixed_p, ref_p = np.array(x0).T, np.array(fixed_p).T, np.array(ref_p).T
        p = np.concatenate((fixed_p, ref_p, theta.T), axis=0)

        # Forward pass through solver
        soln_batch = self.pi_solvers_train(x0=x0, p=p, lbg=lbg, ubg=ubg)
        z = cs.vertcat(soln_batch['x'], soln_batch['lam_g'])
        action_batch = opt_act_fn(soln_batch['x']).full().T

        # Post-processing the solution
        nabla_pi_ref_batch, nabla_pi_theta_batch, optimal_batch = [], [], []
        if compute_sensitivities:
            rkkt_norm_batch, dpi_cs = self.all_solvers_train(z, fixed_p, ref_p, theta.T)
            optimal_batch = rkkt_norm_batch.full() < 1e-3
            nabla_pi_ref_batch, nabla_pi_theta_batch = [], []
            # dpi_cs = dpi_fn_train(optimal_batch, z, fixed_p, ref_p, theta.T).full()
            for i in range(obs_batch.shape[0]):
                # nabla_pi_ref_batch.append(dpi_cs[:ref_p.shape[0], 2 * i: 2 * (i + 1)].T)
                # nabla_pi_theta_batch.append(dpi_cs[ref_p.shape[0]:, 2 * i: 2 * (i + 1)].T)
                nabla_pi_theta_batch.append(
                    int(optimal_batch[0, i]) * dpi_cs[:, self.model.nu * i: self.model.nu * (i + 1)].T
                )
        else:
            # optimal_batch = [True] * obs_batch.shape[0]
            rkkt_norm_batch = self.rkkt_norm_fns_train(z, fixed_p, ref_p, theta.T)
            optimal_batch = rkkt_norm_batch.full() < 1e-3
            nabla_pi_ref_batch = [np.zeros((self.model.nu, ref_p.shape[0]))] * obs_batch.shape[0]
            nabla_pi_theta_batch = [np.zeros((self.model.nu, theta.shape[1]))] * obs_batch.shape[0]
        return action_batch, nabla_pi_ref_batch, nabla_pi_theta_batch, optimal_batch

    def get_parallel_solver(self, n_solvers):
        pi_solvers = self.solver_dict['solver'].map(n_solvers, 'thread')
        rkkt_norm_fns = self.solver_dict['rkkt_norm_fn'].map(n_solvers, 'thread')
        # rkkt_fn = cs.Function.load('rkkt_fn.casadi')
        # rkkt_solvers = rkkt_fn.map(n_solvers, 'thread')
        dpi_fns = self.solver_dict['dpi_fn'].map(n_solvers, 'thread')
        # dPi_fn = cs.Function.load('dPi_fn.casadi')
        # dpi_solvers = dPi_fn.map(n_solvers, 'thread')
        all_fns = self.solver_dict['all_fn'].map(n_solvers, 'thread')
        return pi_solvers, rkkt_norm_fns, dpi_fns, all_fns


class SACBuffer(object):
    '''Storage for a batch of episodes during training.

    Attributes:
        max_length (int): maximum length of episode.
        batch_size (int): number of episodes per batch.
        scheme (dict): describes shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    '''

    def __init__(self, obs_space, act_space, max_size, batch_size):
        super().__init__()
        self.max_size = max_size
        self.batch_size = batch_size
        N = max_size
        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n
        self.scheme = {
            'obs': {'vshape': (N, *obs_dim)},
            'next_obs': {'vshape': (N, *obs_dim)},
            'act': {'vshape': (N, act_dim)},
            'rew': {'vshape': (N, 1)},
            'mask': {'vshape': (N, 1), 'init': np.ones},
            'info': {'vshape': (N,), 'dtype': object, 'init': np.empty},
            # 'optimal': {'vshape': (N, 1)},
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocates space for containers.'''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            # self.__dict__[k] = deque([], maxlen=self.max_size)
            # if k in ['info', 'results_dict']:
            #     self.__dict__[k] = deque([], maxlen=info['vshape'])
            # else:
            vshape = info['vshape']
            dtype = info.get('dtype', np.float32)
            init = info.get('init', np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)
        self.pos = 0
        self.buffer_size = 0

    def __len__(self):
        '''Returns current size of the buffer.'''
        return self.buffer_size

    def state_dict(self):
        '''Returns a snapshot of current buffer.'''
        state = dict(
            pos=self.pos,
            buffer_size=self.buffer_size,
        )
        for k in self.scheme:
            v = self.__dict__[k]
            state[k] = v
        return state

    def load_state_dict(self, state):
        '''Restores buffer from previous state.'''
        for k, v in state.items():
            self.__dict__[k] = v

    def push(self, batch):
        '''Inserts transition step data (as dict) to storage.'''
        k = list(batch.keys())[0]
        n = batch[k].shape[0]

        for k, v in batch.items():
            # assert k in self.keys
            if k == 'info':
                # v should be list-like length n (e.g., list of dicts)
                assert len(v) == n, "info length mismatch"
                if self.pos + n <= self.max_size:
                    self.info[self.pos:self.pos + n] = v
                else:
                    remain_n = self.pos + n - self.max_size
                    self.info[self.pos:self.max_size] = v[:-remain_n]
                    self.info[:remain_n] = v[-remain_n:]
                continue

            # if k not in ['info']:
            shape = self.scheme[k]['vshape'][1:]
            dtype = self.scheme[k].get('dtype', np.float32)
            v = np.asarray(v, dtype=dtype).reshape((n,) + shape)
            if self.pos + n <= self.max_size:
                self.__dict__[k][self.pos:self.pos + n] = v
            else:
                # wrap around
                remain_n = self.pos + n - self.max_size
                self.__dict__[k][self.pos:self.max_size] = v[:-remain_n]
                self.__dict__[k][:remain_n] = v[-remain_n:]
            # else:
            #     self.__dict__[k].extend(v)
        if self.buffer_size < self.max_size:
            self.buffer_size = min(self.max_size, self.pos + n)
        self.pos = (self.pos + n) % self.max_size

    def sample(self, batch_size=None, device=None):
        '''Returns data batch.'''
        if not batch_size:
            batch_size = self.batch_size

        indices = np.random.randint(0, len(self), size=batch_size)
        batch, batch_th = {}, {}
        for k, info in self.scheme.items():
            # if k in ['info']:
            #     batch[k] = [self.__dict__[k][i] for i in indices]
            if k == 'info':
                batch[k] = self.info[indices].tolist()
            else:
                shape = info['vshape'][1:]
                # batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
                # batch_th[k] = self.__dict__[k].reshape(-1, *shape)[indices]
                v = self.__dict__[k].reshape(-1, *shape)[indices]
                batch[k] = v
                if device is None:
                    batch_th[k] = torch.as_tensor(v)
                else:
                    batch_th[k] = torch.as_tensor(v, device=device)
            # data = list(self.__dict__[k])
            # batch[k] = [data[i] for i in indices]
        # for k, v in batch.items():
        #     if k not in ['info', 'results_dict']:
        #         batch_th[k] = torch.as_tensor(np.array(v), dtype=torch.float32, device=device)
        return batch, batch_th


# -----------------------------------------------------------------------------------
#                   Misc
# -----------------------------------------------------------------------------------


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


def _create_semi_definite_matrix(n):
    # U = cs.SX.sym("U", cs.Sparsity.lower(n))
    # u = cs.vertcat(*U.nonzeros())
    # W_upper = cs.Function("Lower_tri_W", [u], [U])
    # np = int(n * (n + 1) / 2)
    # p = cs.MX.sym("p", np)
    # W = W_upper(p)
    # WW = W.T @ W

    n_param = n
    P = cs.MX.sym('P', n)
    W = cs.diag(P)
    # WW = cs.sqrt(W.T @ W)
    return W, P, n_param


def soft_update(source, target, tau):
    '''Synchronizes target networks with exponential moving average.'''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(source, target):
    '''Synchronizes target networks by copying over parameters directly.'''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
