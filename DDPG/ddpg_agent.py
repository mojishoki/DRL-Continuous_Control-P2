import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperparameters import *
from ddpg_model import *
from ReplayBuffer import *
from OUNoise import *

class ddpg_agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, gamma= GAMMA,
                 tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY,
                 update_every=UPDATE_EVERY, learn_num=LEARN_NUM, random_seed=SEED):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            batch_size (int): batch size for replay buffer default at 128
            buffer_size (int): buffer size for replay buffer default at 1e6
            gamma: discount factor, set at 0.99
            tau (float): soft-update hyper parameter default at 1e-3
            lr_actor: lr for actor default at 1e-3
            lr_critic: lr for critic default at 1e-3
            weight_decay: weight decay for adam optimizer default at 0
            update_every: update every timesteps default at 20
            learn_num: number of times to update at update_every timesteps default at 10
            random_seed (int): random seed default at 1
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.gamma=gamma
        self.tau=tau
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.weight_decay=weight_decay
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
         
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, random_seed)
        
        #update_every
        self.update_every=update_every
        self.learn_num=learn_num
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                #loop here to update many times for a update_every time_step
                for _ in range(self.learn_num):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True, watch_phase=False):
        """Returns actions for given state as per current policy. 
        `add_noise`: For OUNoise to apply additively
        `watch_phase`: For OUNoise to apply need watch_phase=False, set True if want to watch agent play without noise"""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        self.t_step = (self.t_step + 1) % self.update_every
        if not watch_phase and add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Loss of actor is simply negative the action-value (which needs to be maximized), i.e. output of critic.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)  

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)