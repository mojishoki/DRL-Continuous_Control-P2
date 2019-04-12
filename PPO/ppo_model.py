import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
    
class ActorCriticPPO(nn.Module):
    """ActorCritic PPO Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=512):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorCriticPPO, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor_fc1 = nn.Linear(state_size, fc1_units)
        self.actor_bn1 = nn.BatchNorm1d(fc1_units)
        self.actor_fc2 = nn.Linear(fc1_units, fc2_units)
        self.actor_fc3 = nn.Linear(fc2_units, action_size)
        self.actor_std = nn.Parameter(torch.ones(action_size))
        
        self.critic_fc1 = nn.Linear(state_size, fc1_units)
        self.critic_bn1 = nn.BatchNorm1d(fc1_units)
        self.critic_fc2 = nn.Linear(fc1_units, fc2_units)
        self.critic_fc3 = nn.Linear(fc2_units, 1)
        #we found no need to do Xavier-He initialization
#         self.reset_parameters([self.critic_fc1,self.critic_fc2,self.critic_fc3])
#         self.reset_parameters([self.actor_fc1,self.actor_fc2,self.actor_fc3])
    def reset_parameters(self,layers):
        for i in range(len(layers)-1):
            layers[i].weight.data.uniform_(*hidden_init(layers[i]))
        layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, actions = None):
        """Build an actor (policy) network that maps states -> actions."""
        state=torch.tensor(state).to(device).float()
        x = F.relu(self.actor_fc1(state))
#         x = F.relu(self.actor_bn1(self.actor_fc1(state))) #using batch or not
        x = F.relu(self.actor_fc2(x))
        mean = torch.tanh(self.actor_fc3(x))
        dist = torch.distributions.Normal(mean,self.actor_std)
        if type(actions)==type(None):
            actions = torch.clamp(dist.sample(),-1,1)
        else:
            actions=torch.tensor(actions).to(device)
        log_probs = dist.log_prob(actions).sum(-1).unsqueeze(-1)
        
        v = F.relu(self.critic_fc1(state))
#         v = F.relu(self.critic_bn1(self.critic_fc1(state))) #using batch or not
        v = self.critic_fc3(F.relu(self.critic_fc2(v)))

        return [actions, log_probs, v, dist.entropy()] # of shape b*(4,1,1,4)
                #entropy given only by std for normal distributions  (did not do F.softplus)
                #log_probs is also where mean and std appear