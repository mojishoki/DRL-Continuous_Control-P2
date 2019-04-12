import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def states_actions_to_prob(policy, states, actions):
    """ Taking in trajectories and applying policy on it """
    policy.train()
    #need to reshape as shape is max_t*num_agents*sth , need batch dim to be max_t*num_agents
    states, actions= np.array(states),np.array(actions) 
    states_input = states.reshape(-1,states.shape[-1])
    actions_input = actions.reshape(-1,actions.shape[-1])
    output=policy(states_input,actions_input)
    output[0]=output[0].view(*actions.shape)
    output[1]=output[1].view(actions.shape[0:2]).unsqueeze(dim=2)
    output[2]=output[2].view(actions.shape[0:2]).unsqueeze(dim=2)
    output[3]=output[3].view(*actions.shape)
    return output #shape max_t*num_ag*(4,1,1,4)

def surrogate(policy, old_log_probs, states, actions,returns,advantage,epsilon=0.2, c1 = 0.5, beta1=0.01,
                     beta2=0.01,delta= 0.1, loss_kind="entropy"):
    """ 
    Surrogate function implementing the loss function. Takes as input:
        policy: the PPO ActorCritic policy model
        old_log_probs: the log probs of trajectories collected
        states: the states of trajectories collected
        actions: the actions of trajectories collected
        returns: the future returns (discounted rewards sum) of trajectories collected  
        advantage: the advantage computed using old_values and returns
    """
       
    actions, new_log_probs, new_v,  new_entropy = states_actions_to_prob(policy,states, actions) 
    ratio = (new_log_probs-old_log_probs).exp()
    ratio_clamped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    #computing Lsur_clipped
    Lsur = advantage*ratio
    Lsur_clamped = advantage*ratio_clamped
    Lsur_clipped = torch.min(Lsur,Lsur_clamped)
    #computing value loss and entropy and KL distances
    value_loss=(new_v-returns)**2
    new_entropy=torch.mean(new_entropy,dim=2,keepdim=True)
    new_policy_entropy = -(new_log_probs.exp()*new_log_probs)
    new_old_policy_KL = (new_log_probs.exp()*(new_log_probs-old_log_probs))
    if loss_kind == "simplest":
        return torch.mean(Lsur_clipped-c1*value_loss)
    if loss_kind == "entropy":
        return torch.mean(Lsur_clipped-c1*value_loss+beta1*new_policy_entropy+beta2*new_entropy)
    if loss_kind == "KL_entropy_approximate":
        return torch.mean(Lsur_clipped-c1*value_loss+beta1*new_policy_entropy-delta*new_old_policy_KL)
    if loss_kind == "entropy_exact":
        return torch.mean(Lsur_clipped-c1*value_loss+beta2*new_entropy)
    if loss_kind == "entropy_approximate":
        return torch.mean(Lsur_clipped-c1*value_loss+beta1*new_policy_entropy)
    if loss_kind == "KL_approximate":
        return torch.mean(Lsur_clipped-c1*value_loss-delta*new_old_policy_KL)