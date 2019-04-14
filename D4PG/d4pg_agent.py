import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperparameters import *
from d4pg_model import *
from ReplayBuffer import *
from OUNoise import *

class d4pg_agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, gamma= GAMMA,
                 tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY,
                 update_every=UPDATE_EVERY, learn_num=LEARN_NUM, eps=EPSILON, eps_decay=EPSILON_DECAY,rollout_length=ROLLOUT_LENGTH,
                 atoms=ATOMS, V_max=V_MAX, V_min=V_MIN, hard_update=HARD_UPDATE, soft_update=True, eps_gauss=False, random_seed=SEED):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            batch_size (int): batch size for replay buffer default at 128
            buffer_size (int): buffer size for replay buffer default at 1e6
            gamma: discount factor, set at 0.99
            tau (float): soft-update hyper parameter default at 1e-3
            lr_actor: lr for actor default at 1e-4
            lr_critic: lr for critic default at 1e-3
            weight_decay: weight decay for adam optimizer default at 1e-5
            update_every: update every timesteps default at 20
            learn_num: number of times to update at update_every timesteps default at 10
            eps: the epsilon used for adding gaussian noise with mean 0 and std 1
            eps_decay: *linear* rate of decay of epsilon (subtracted after each update), in d4pg paper it is not decayed so you might want it to be one, default at 1e-6
            rollout_length (int): length of each trajectory used for training, default 5
            atoms (int): number of atoms to use in the categorical distribution, default 51
            V_max,V_min: the interval [V_min,V_max] is where the atoms are put (equidistant)
            hard_update: how many steps before doing a hard update on the target networks
            soft_update: if True will use soft_update with `tau`, if False will use hard update at each hard_update many steps, d4pg paper
            used hard_update
            eps_gauss: Whether to use OUNoise or use gaussian (d4pg uses guassian noise)
            random_seed (int): random seed default at 1
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        #hyperparams
        self.batch_size=batch_size
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.rollout_length=rollout_length
        self.gamma=gamma
        self.tau=tau
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.weight_decay=weight_decay
        #the atoms
        self.num_atoms=atoms
        self.V_max=V_max
        self.V_min=V_min
        self.delta=(self.V_max-self.V_min)/(self.num_atoms-1)
        self.atoms=torch.tensor([self.delta*i+self.V_min for i in range(self.num_atoms)])
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, atoms=self.num_atoms).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, atoms=self.num_atoms).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
         
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, random_seed, self.rollout_length)
        
        #update_every
        self.update_every=update_every
        self.learn_num=learn_num
        self.soft=soft_update
        self.t_step = 0
        self.t_hard=0
        self.hard= hard_update
        
        #gaussian noise
        self.eps=eps
        self.eps_decay=eps_decay
        self.eps_gauss=eps_gauss
        self.means=torch.tensor([0,0,0,0]).float()
        self.stds=torch.tensor([1,1,1,1]).float()
        self.gauss=torch.distributions.Normal(self.means, self.stds)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                #update learn_num times for update_every time_step
                for _ in range(self.learn_num):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True, watch_phase=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        self.t_step = (self.t_step + 1) % self.update_every
        if self.soft==False:
            self.t_hard = (self.t_hard + 1) % self.hard
        if not watch_phase and add_noise:
            if self.eps_gauss:
                action += self.gauss.sample()*self.eps
            else:
                action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """ resetting the OUNoise """
        self.noise.reset()
    
    def project_dist(self,dist,returns_tmp,gamma):
        """ 
        Projecting the categorical distribution of target as shown in "A Distributional Perspective on Reinforcement Learning" paper 
        """
        proj_atoms=torch.zeros((self.num_atoms,self.batch_size)).to(device)
        dist=dist.permute([1,0]) #num_atoms*B
        for i in range(self.num_atoms):
            #size B 
            to_proj=returns_tmp.squeeze()+gamma**self.rollout_length * self.atoms[i]
            proj_atoms[i]=torch.clamp(torch.tensor(to_proj),self.V_min,self.V_max)
            
#         direct implementation of the eq (7) in Distributional paper:
#         new_dist=torch.zeros_like(dist).to(device)
#         for i in range(self.num_atoms):
#             new_dist[i]=(torch.clamp(1-torch.abs(proj_atoms-self.atoms[i])/self.delta,0,1)*dist).sum(dim=0)

#         implementation of algorithm 1 
#         size 1* num_atoms * num_atoms*B = 1*B
        new_dist=torch.zeros_like(dist).to(device)
        #here, distributional paper uses the best_value instead of dist[j] which would mean taking expectation of dist; 
        #but implementating dist[j] seems to give better results!
        for j in range(self.num_atoms):    
            bj=torch.zeros(self.batch_size).to(device)
            bj=(proj_atoms[j]-self.V_min)/self.delta
            #shape B
            u=torch.ceil(bj).long()
            l=torch.floor(bj).long()
            new_dist[l]+=dist[j]*(u.float()-bj)
            new_dist[u]+=dist[j]*(bj-l.float())
        return new_dist.permute([1,0])
        
        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():            
            actions_next = self.actor_target(next_states[self.rollout_length-1])
            Z_target_next = self.critic_target(next_states[self.rollout_length-1], actions_next)
            # Compute returns_tmp size B*1
            returns_tmp = np.array([rewards[i]*(gamma**i) for i in range(self.rollout_length)]).sum(axis=0)
            #dimension B*self.num_atoms
            Y_target = returns_tmp + (gamma**self.rollout_length * Z_target_next * (1 - dones[self.rollout_length-1]))
            Y_target = self.project_dist(Y_target,returns_tmp,gamma)

        # Compute critic loss
        self.critic_local.train()
        Z_expected = self.critic_local(states[0], actions[0])
        #to perform prioritized here need to add weight=BUFFER_SIZE/p with p the probability
        critic_loss = -(Y_target*torch.log(Z_expected+1e-10)).sum(dim=1).mean()
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states[0])
        #basically it's about maximizing the Q(s,action_provided_by_actor) hence the negative sign, B*atoms matmul atoms*1
        actor_loss = -self.critic_local(states[0], actions_pred).matmul(self.atoms.to(device)).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.soft_update:
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)
        elif self.t_hard==0:
            self.hard_update(self.critic_local, self.critic_target)
            self.hard_update(self.actor_local, self.actor_target)
        if self.eps_gauss:
            self.eps=self.eps-self.eps_decay
            

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
    def hard_update(self,local_model,target_model):
        """ Copy the local_model state dict into that of target_model """
        target_model.load_state_dict(local_model.state_dict())