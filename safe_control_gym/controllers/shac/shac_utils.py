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
                 clip_param=0.2,
                 target_kl=0.01,
                 entropy_coef=0.01,
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
        self.clip_param = clip_param
        self.target_kl = target_kl
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

    def compute_policy_loss(self,
                            batch
                            ):
        '''Returns policy loss(es) given batch of data.'''
        obs, act, logp_old, adv = batch['obs'], batch['act'], batch['logp'], batch['adv']
        dist, logp = self.ac.actor(obs, act)
        # Policy.
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()
        # Entropy.
        entropy_loss = -dist.entropy().mean()
        # KL/trust region.
        approx_kl = (logp_old - logp).mean()
        return policy_loss, entropy_loss, approx_kl

    def compute_value_loss(self,
                           batch
                           ):
        '''Returns value loss(es) given batch of data.'''
        obs, ret, v_old = batch['obs'], batch['ret'], batch['v']
        v_cur = self.ac.critic(obs)
        if self.use_clipped_value:
            v_old_clipped = v_old + (v_cur - v_old).clamp(-self.clip_param, self.clip_param)
            v_loss = (v_cur - ret).pow(2)
            v_loss_clipped = (v_old_clipped - ret).pow(2)
            value_loss = 0.5 * torch.max(v_loss, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        return value_loss

    def update(self,
               rollouts,
               actor_loss,
               device='cpu'
               ):
        '''Updates model parameters based on current training batch.'''
        results = defaultdict(list)
        num_mini_batch = rollouts.max_length * rollouts.batch_size // self.mini_batch_size
        # assert if num_mini_batch is not 0
        assert num_mini_batch != 0, 'num_mini_batch is 0'
        n_updates = 0

        # actor update
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()
        results['actor_loss'].append(actor_loss.item())
        
        for _ in range(self.opt_epochs):
            v_loss_epoch = 0
            for batch in rollouts.sampler(self.mini_batch_size, device):
                # Critic update.
                value_loss = self.compute_value_loss(batch)
                self.critic_opt.zero_grad()
                value_loss.backward()
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

    def forward(self,
                obs,
                ):
        dist = self.dist_fn(self.pi_net(obs))
        return dist


class MLPCritic(nn.Module):
    '''Critic MLP model.'''

    def __init__(self,
                 obs_dim,
                 hidden_dims,
                 activation
                 ):
        super().__init__()
        self.v_net = MLP(obs_dim, 1, hidden_dims, activation)

    # def forward(self,
    #             obs
    #             ):
    #     v = self.v_net(obs)
    #     return v
    
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
             obs
             ):
        dist = self.actor(obs)
        action = dist.rsample()
        v = self.critic(obs)
        return action, v

    def act(self,
            obs,
            extra_info=False
            ):
        dist = self.actor(obs)
        action = dist.mode()
        v = self.critic(obs)
        if extra_info:
            return action.cpu().numpy(), v.cpu().numpy()
        return action.cpu().numpy()


class SHACBuffer(object):
    '''Storage for a batch of episodes during training.

    Attributes:
        max_length (int): maximum length of episode.
        batch_size (int): number of episodes per batch.
        scheme (dict): describs shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    '''

    def __init__(self,
                 obs_space,
                 act_space,
                 max_length,
                 batch_size
                 ):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        T, N = max_length, batch_size
        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n
        self.scheme = {
            'obs': {
                'vshape': (T, N, *obs_dim)
            },
            'act': {
                'vshape': (T, N, act_dim)
            },
            'rew': {
                'vshape': (T, N, 1)
            },
            'mask': {
                'vshape': (T, N, 1),
                'init': np.ones
            },
            'v': {
                'vshape': (T, N, 1)
            },
            'ret': {
                'vshape': (T, N, 1)
            },
            'terminal_v': {
                'vshape': (T, N, 1)
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocates space for containers.'''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            vshape = info['vshape']
            dtype = info.get('dtype', np.float32)
            init = info.get('init', np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)
        self.t = 0

    def push(self,
             batch
             ):
        '''Inserts transition step data (as dict) to storage.'''
        for k, v in batch.items():
            assert k in self.keys
            shape = self.scheme[k]['vshape'][1:]
            dtype = self.scheme[k].get('dtype', np.float32)
            v_ = np.asarray(deepcopy(v), dtype=dtype).reshape(shape)
            self.__dict__[k][self.t] = v_
        self.t = (self.t + 1) % self.max_length

    def get(self,
            device='cpu'
            ):
        '''Returns all data.'''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]
            data = self.__dict__[k].reshape(-1, *shape)
            batch[k] = torch.as_tensor(data, device=device)
        return batch

    def sample(self,
               indices
               ):
        '''Returns partial data.'''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]
            batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
        return batch

    def sampler(self,
                mini_batch_size,
                device='cpu',
                drop_last=True
                ):
        '''Makes sampler to loop through all data.'''
        total_steps = self.max_length * self.batch_size
        sampler = random_sample(np.arange(total_steps), mini_batch_size, drop_last)
        for indices in sampler:
            batch = self.sample(indices)
            batch = {
                k: torch.as_tensor(v, device=device) for k, v in batch.items()
            }
            yield batch


def random_sample(indices,
                  batch_size,
                  drop_last=True
                  ):
    '''Returns index batches to iterate over.'''
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(
        -1, batch_size)
    for batch in batches:
        yield batch
    if not drop_last:
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]


def compute_shac_returns_and_actor_loss(act_dim, 
                                        act_v_list,
                                        rews,
                                        vals,
                                        masks,
                                        terminal_vals=0,
                                        last_val=0,
                                        last_val_grad=0,
                                        gamma=0.99,
                                        device='cpu'
                                        ):
    '''Useful for policy-gradient algorithms.'''
    T, N = rews.shape[:2]
    rets = np.zeros((T, N, 1))
    ret = last_val
    # Compensate for time truncation.
    rews_eff = rews +  gamma * terminal_vals
    # Loss graph for SHAC.
    lambda_next = last_val_grad
    actor_loss = 0.0
    masks_th = torch.as_tensor(masks, dtype=torch.float32, device=device)
    # Cumulative discounted sums.
    for i in reversed(range(T)):
        ret = rews_eff[i] + gamma * masks[i] * ret
        rets[i] = deepcopy(ret)

        # Create graph for SHAC.
        action_t, terminal_v_grad_t, diff_info_t = act_v_list[i]
        lambda_t = torch.zeros_like(lambda_next)
        q_t_all = torch.zeros((N, act_dim), dtype=torch.float32, device=device)
        for j in range(N):
            diff = diff_info_t[j]
            rew_x = torch.as_tensor(diff['rew_x'], dtype=torch.float32, device=device)
            rew_u = torch.as_tensor(diff['rew_u'], dtype=torch.float32, device=device)
            nx_x = torch.as_tensor(diff['nx_x'], dtype=torch.float32, device=device)
            nx_u = torch.as_tensor(diff['nx_u'], dtype=torch.float32, device=device)
            obs_dim = rew_x.numel()
            lambda_eff = masks_th[i, j] * lambda_next[j] + terminal_v_grad_t[j]
            q_t = rew_u + gamma * (lambda_eff[:obs_dim] @ nx_u)
            lambda_t[j, :obs_dim] = rew_x + gamma * (lambda_eff[:obs_dim] @ nx_x)
            q_t_all[j] = q_t

            # ret_graph[i, j] = info_t['n'][j]['diff_sim_info']['rew_u'] + gamma * (lambda_next[j] @ info_t['n'][j]['diff_sim_info']['nx_u'])
            # lambda_t[j] = info_t['n'][j]['diff_sim_info']['rew_x'] + gamma * (lambda_next[j] @ info_t['n'][j]['diff_sim_info']['nx_x'])
            # lambda_t[j] += 0.0 # add policy grad
        actor_loss -= (action_t * q_t_all.detach()).sum(dim=1).mean()
        lambda_next = lambda_t
    actor_loss = actor_loss / T
    return rets, actor_loss
