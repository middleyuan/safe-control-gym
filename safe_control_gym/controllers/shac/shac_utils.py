'''SHAC utilities.'''

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.math_and_models.distributions import Categorical, Normal
from safe_control_gym.math_and_models.neural_networks import MLP


class SHACAgent:
    '''A SHAC class that encapsulates models, optimizers and update functions.'''

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dim=64,
                 use_clipped_value=False,
                 entropy_coef=0.0,
                 exploration_init=-0.5,
                 actor_lr=0.0003,
                 critic_lr=0.001,
                 opt_epochs=10,
                 mini_batch_size=64,
                 activation='tanh',
                 **kwargs
                 ):
        # Parameters.
        self.obs_space = obs_space
        self.act_space = act_space
        self.use_clipped_value = use_clipped_value
        self.entropy_coef = entropy_coef
        self.opt_epochs = opt_epochs
        self.mini_batch_size = mini_batch_size
        self.activation = activation
        # Model.
        self.ac = MLPActorCritic(obs_space,
                                 act_space,
                                 hidden_dims=[hidden_dim] * 2,
                                 activation=self.activation,
                                 exploration_init=exploration_init)
        # Optimizers.
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), critic_lr)

    def to(self,
           device
           ):
        '''Puts agent to device.'''
        self.ac.to(device)

    def train(self):
        '''Sets training mode.'''
        self.ac.train()

    def eval(self):
        '''Sets evaluation mode.'''
        self.ac.eval()

    def state_dict(self):
        '''Snapshots agent state.'''
        return {
            'ac': self.ac.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict()
        }

    def load_state_dict(self,
                        state_dict
                        ):
        '''Restores agent state.'''
        self.ac.load_state_dict(state_dict['ac'])
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])

    def compute_value_loss(self,
                           batch
                           ):
        '''Returns value loss(es) given batch of data.'''
        obs, ret = batch['obs'], batch['ret']
        v_cur = self.ac.critic(obs)
        value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        return value_loss

    def update(self,
               rollouts,
               actor_loss,
               entropy_loss,
               device='cpu'
               ):
        '''Updates model parameters based on current training batch.'''
        results = defaultdict(list)
        num_mini_batch = rollouts.max_length * rollouts.batch_size // self.mini_batch_size
        # assert if num_mini_batch is not 0
        assert num_mini_batch != 0, 'num_mini_batch is 0'

        # actor update
        self.actor_opt.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()
        results['actor_loss'].append(actor_loss.item())
        results['entropy_loss'].append(entropy_loss.item())
        
        for _ in range(self.opt_epochs):
            v_loss_epoch, n_updates = 0, 0
            for batch in rollouts.sampler(self.mini_batch_size):
                # Critic update.
                value_loss = self.compute_value_loss(batch)
                self.critic_opt.zero_grad()
                value_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), max_norm=10.0)
                self.critic_opt.step()
                # logging
                v_loss_epoch += value_loss.item()
                n_updates += 1
            results['value_loss'].append(v_loss_epoch / max(n_updates, 1))
        results = {k: sum(v) / len(v) for k, v in results.items()}
        return results


class MLPActor(nn.Module):
    '''Actor MLP model.'''

    def __init__(self,
                 obs_dim,
                 action_space,
                 hidden_dims,
                 activation,
                 discrete=False,
                 exploration_init=-0.5
                 ):
        super().__init__()
        act_dim = action_space.shape[0]
        self.pi_net = MLP(obs_dim, act_dim, hidden_dims, activation)
        # Construct output action distribution.
        self.logstd = nn.Parameter(exploration_init * torch.ones(act_dim))
        self.dist_fn = lambda x: Normal(x, self.logstd.exp())
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self,
                obs,
                ):
        dist = self.dist_fn(self.pi_net(obs))
        return dist

    def squash(self, raw_action):
        return torch.tanh(raw_action) * self.action_scale + self.action_bias

    def log_prob_from_raw_action(self, dist, raw_action):
        """Log-probability corrected for tanh squashing and action rescaling."""
        y_t = torch.tanh(raw_action)
        logp = dist.log_prob(raw_action)
        logp -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6).sum(-1, keepdim=True)
        return logp


class MLPCritic(nn.Module):
    '''Critic MLP model.'''

    def __init__(self,
                 obs_dim,
                 hidden_dims,
                 activation
                 ):
        super().__init__()
        self.v_net = MLP(obs_dim, 1, hidden_dims, activation)
    
    def forward(self, obs, return_grad=False, create_graph=False):
        """
        Args:
            obs: [B, obs_dim]
            return_grad: if True, also return dV/dobs
            create_graph: True if you need higher-order gradients later

        Returns:
            v: [B, 1]
            v_grad: [B, obs_dim] if return_grad=True
        """
        if not return_grad:
            return self.v_net(obs)

        # Important: obs must require grad
        obs = obs.detach().clone().requires_grad_(True)

        v = self.v_net(obs)  # [B, 1]

        v_grad = torch.autograd.grad(
            outputs=v.sum(),
            inputs=obs,
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True,
        )[0]

        return v.detach(), v_grad.detach()


class MLPActorCritic(nn.Module):
    '''Model for the actor-critic agent.

    Attributes:
        actor (MLPActor): policy network.
        critic (MLPCritic): value network.
    '''

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dims=(64, 64),
                 activation='tanh',
                 exploration_init=-0.5
                 ):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
            discrete = False
        else:
            act_dim = act_space.n
            discrete = True
        # Policy.
        self.actor = MLPActor(obs_dim, act_space, hidden_dims, activation, discrete, exploration_init)
        # Value function.
        self.critic = MLPCritic(obs_dim, hidden_dims, activation)

    def step(self,
             obs,
             extra_info=False
             ):
        dist = self.actor(obs)
        raw_action = dist.rsample()
        action = self.actor.squash(raw_action)
        if extra_info:
            logp = self.actor.log_prob_from_raw_action(dist, raw_action)
            return action, logp
        return action

    def act(self,
            obs,
            extra_info=False
            ):
        dist = self.actor(obs)
        action = self.actor.squash(dist.mode())
        if extra_info:
            return action.cpu().numpy()
        return action.cpu().numpy()
    
    def actor_vjp_state(self, obs, q_t, state_dim=None, action=None):
        """
        Computes q_t @ d pi(obs) / d obs efficiently.

        Args:
            obs: [N, obs_dim]
            q_t: [N, 1, act_dim] or [N, act_dim]
            state_dim: optional, only keep first state_dim components

        Returns:
            q_pi_x: [N, 1, state_dim] if q_t was [N, 1, act_dim]
                    or [N, state_dim] if q_t was [N, act_dim]
        """
        obs_req = obs.detach().clone().requires_grad_(True)

        mean = self.actor.pi_net(obs_req)  # [N, act_dim]
        if action is None:
            policy_action = self.actor.squash(mean)
            scalar = (q_t.detach() @ policy_action.unsqueeze(-1)).sum()
        else:
            normalized_action = (action.detach() - self.actor.action_bias) / self.actor.action_scale
            normalized_action = normalized_action.clamp(-1.0, 1.0)
            squash_grad = self.actor.action_scale * (1.0 - normalized_action.pow(2))
            weighted_q = q_t.detach().squeeze(1) if q_t.dim() == 3 else q_t.detach()
            scalar = (weighted_q * squash_grad * mean).sum()

        q_pi_x = torch.autograd.grad(
            outputs=scalar,
            inputs=obs_req,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]  # [N, obs_dim]

        if state_dim is not None:
            q_pi_x = q_pi_x[:, :state_dim]
        return q_pi_x.detach().unsqueeze(1) if q_t.dim() == 3 else q_pi_x.detach()


class SHACBuffer(object):
    def __init__(self, obs_space, act_space, max_length, batch_size, device="cpu"):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        T, N = max_length, batch_size
        obs_dim = obs_space.shape

        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n

        self.scheme = {
            "obs": {"vshape": (T, N, *obs_dim)},
            "act": {"vshape": (T, N, act_dim)},
            "rew": {"vshape": (T, N, 1)},
            "mask": {"vshape": (T, N, 1), "init": "ones"},
            "ret": {"vshape": (T, N, 1)},
            "terminal_v": {"vshape": (T, N, 1)},
        }

        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        for k, info in self.scheme.items():
            vshape = info["vshape"]
            init = info.get("init", "zeros")

            if init == "ones":
                self.__dict__[k] = torch.ones(
                    vshape, dtype=torch.float32, device=self.device
                )
            else:
                self.__dict__[k] = torch.zeros(
                    vshape, dtype=torch.float32, device=self.device
                )

        self.t = 0

    def push(self, batch):
        for k, v in batch.items():
            assert k in self.keys

            shape = self.scheme[k]["vshape"][1:]

            if torch.is_tensor(v):
                v_t = v.detach().to(self.device, dtype=torch.float32).reshape(shape)
            else:
                v_t = torch.as_tensor(
                    v, dtype=torch.float32, device=self.device
                ).reshape(shape)

            self.__dict__[k][self.t].copy_(v_t)

        self.t = (self.t + 1) % self.max_length

    def get(self):
        batch = {}
        for k, info in self.scheme.items():
            shape = info["vshape"][2:]
            batch[k] = self.__dict__[k].reshape(-1, *shape)
        return batch

    def sample(self, indices):
        batch = {}

        for k, info in self.scheme.items():
            shape = info["vshape"][2:]
            data = self.__dict__[k].reshape(-1, *shape)
            batch[k] = data[indices]

        return batch

    def sampler(self, mini_batch_size, drop_last=True):
        total_steps = self.max_length * self.batch_size

        indices = torch.randperm(total_steps, device=self.device)

        if drop_last:
            end = total_steps // mini_batch_size * mini_batch_size
            indices = indices[:end]

        for idx in indices.split(mini_batch_size):
            if drop_last and idx.numel() < mini_batch_size:
                continue
            yield self.sample(idx)


def compute_shac_returns_and_actor_loss(
        env,
        diff_sim,
        rews,
        masks,
        terminal_vals=0,
        last_val=0,
        last_val_grad=0,
        gamma=0.99,
        actor_vjp_fn=None,
        device='cpu',
        cs_workers=1,
        entropy_coef=0.0
    ):
    '''Compute returns and the SHAC actor loss with mapped CasADi derivatives.'''
    T, N = rews.shape[:2]
    rets = torch.zeros((T, N, 1), dtype=torch.float32, device=device)
    ret = last_val
    rews_eff = rews + gamma * terminal_vals
    lambda_next = last_val_grad
    actor_loss = 0.0
    masks_th = torch.as_tensor(masks, dtype=torch.float32, device=device).unsqueeze(-1)
    logps = torch.stack([step[6] for step in diff_sim]).to(device)

    # Prepare mapped CasADi derivatives for SHAC loss computation.
    state_dim = env.state_space.shape[0]
    act_dim = env.action_space.shape[0]
    num_transitions = T * N
    reward_fn = env.symbolic.reward_func.map(num_transitions, "thread", cs_workers)
    dnxdx_fn = env.dnxdx_func.map(num_transitions, "thread", cs_workers)
    dnxdu_fn = env.dnxdu_func.map(num_transitions, "thread", cs_workers)
    next_states = np.asarray([step[0] for step in diff_sim], dtype=np.float64)
    obss = torch.stack([step[1] for step in diff_sim]).to(device)
    prev_states = obss[:, :, :state_dim].detach().cpu().numpy()
    actions_np = np.asarray(
        [step[2].detach().cpu().numpy() for step in diff_sim], dtype=np.float64
    )
    x_ref = np.asarray([step[3] for step in diff_sim], dtype=np.float64)
    u_ref = np.asarray([step[4] for step in diff_sim], dtype=np.float64)
    disturbance_dim = env.dnxdx_func.size1_in(2)
    disturbances = np.zeros((num_transitions, disturbance_dim), dtype=np.float64)

    # Reward and dynamics derivatives are independent across time and rollout
    # environments. Evaluate all T * N transitions in three mapped CasADi calls.
    rew_info = reward_fn(
        x=next_states.reshape(num_transitions, state_dim).T,
        u=actions_np.reshape(num_transitions, act_dim).T,
        Xr=x_ref.reshape(num_transitions, state_dim).T,
        Ur=u_ref.reshape(num_transitions, act_dim).T,
        Q=env.Q,
        R=env.R,
    )
    rew_x = torch.as_tensor(
        rew_info["exp_r_x"].full().T.reshape(T, N, state_dim),
        dtype=torch.float32,
        device=device,
    )
    rew_u = torch.as_tensor(
        rew_info["exp_r_u"].full().T.reshape(T, N, act_dim),
        dtype=torch.float32,
        device=device,
    )

    nx_x = dnxdx_fn(
        X=prev_states.reshape(num_transitions, state_dim).T,
        U=actions_np.reshape(num_transitions, act_dim).T,
        d=disturbances.T,
    )["dnxdx"].full()
    nx_u = dnxdu_fn(
        X=prev_states.reshape(num_transitions, state_dim).T,
        U=actions_np.reshape(num_transitions, act_dim).T,
        d=disturbances.T,
    )["dnxdu"].full()
    nx_x = torch.as_tensor(
        nx_x.reshape(state_dim, num_transitions, state_dim)
            .transpose(1, 0, 2)
            .reshape(T, N, state_dim, state_dim),
        dtype=torch.float32,
        device=device,
    )
    nx_u = torch.as_tensor(
        nx_u.reshape(state_dim, num_transitions, act_dim)
            .transpose(1, 0, 2)
            .reshape(T, N, state_dim, act_dim),
        dtype=torch.float32,
        device=device,
    )

    for i in reversed(range(T)):
        ret = rews_eff[i] + gamma * masks[i] * ret
        rets[i].copy_(ret)

        action_t = diff_sim[i][2]
        # terminal_v_grad_t = torch.tensor(diff_sim[i][5], dtype=torch.float32, device=device)
        terminal_v_grad_t = diff_sim[i][5]
        lambda_eff = masks_th[i] * lambda_next + terminal_v_grad_t
        next_state_grad = rew_x[i].unsqueeze(1) + gamma * lambda_eff
        q_t = rew_u[i].unsqueeze(1) + next_state_grad @ nx_u[i]
        lambda_t = next_state_grad @ nx_x[i]

        if actor_vjp_fn is not None:
            lambda_t += actor_vjp_fn(
                obss[i], q_t, state_dim=state_dim, action=action_t
            )

        actor_loss -= (action_t.unsqueeze(1) * q_t.detach()).sum(dim=1).mean()
        lambda_next = lambda_t

    actor_loss = actor_loss / T
    entropy_loss = logps.mean()
    actor_loss = actor_loss + entropy_coef * entropy_loss
    return rets, actor_loss, entropy_loss
