import progressbar as pb
import numpy as np
import torch
from ppo_surrogate import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ppo_train(agent,env,brain_name,episode = 500,max_t = 1000, SGD_epoch = 30, target_score = 30 , discount = .99, eta=0.95, epsilon = 0.2, c1 = 0.5, beta = .01, delta=0.1, clip=0.2, loss_kind="entropy_exact"):
    """
        epsiode: number of episodes to run
        max_t: max number of steps; set at no more than 1000 to avoid terminal states issues (did not implement `dones`)
        SGD_epoch: number of updates using the old policy trajectories
        target_score: score to reach to solve the env
        discount: discount for reward
        eta: the Generalized Advantage Estimation (GAE) hyperparameter usually between 0.92 to 0.98 (if 1., it is just MC)
        epsilon: the clipping ppo hyperparameter
        Loss scalar factors:
        c1: the scalar factor for value_loss default 0.5
        beta: the scalar factor for new policy entropy and also new policy distribution entropy
        delta: the scalar factor for new and old policy KL distance
        clip: clipping the gradients
        Loss type:
        loss_kind: can be either of ["simplest","entropy","entropy_exact","entropy_approximate","KL_approximate","KL_entropy_approximate"]
            simplest: No consideration of entropies or KL
            entropy: Consider both forms of entropies
            entropy_exact: Consider only new policy distribution entropy
            entropy_approximate: Consider only new policy entropy
            KL_approximate: Consider only new and old policy KL distance
            KL_entropy_approximate: merging entropy_approximate and KL_approximate
    """
    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()
    mean_rewards,min_rewards,max_rewards,mavg_rewards=([] for _ in range(4))
    stats=f'Checkpoints/'
    stats+=f'{discount}{eta}{epsilon}{beta}{c1}{delta}{max_t}{SGD_epoch}{loss_kind}{clip}{agent.lr}'
    #preparing the discount and eta_discount arrays to compute future returns and GAE
    discounts=discount**np.arange(max_t)[:,np.newaxis,np.newaxis]
    eta_discounts=(eta*discount)**np.arange(max_t)[:,np.newaxis,np.newaxis]
    for e in range(1,episode+1):
        states, actions,old_log_probs, rewards,v=agent.collect_trajectories(env,brain_name,max_t=max_t,n_traj=1)
        total_rewards = np.sum(rewards, axis=0)
        v=np.array(v)
        rewards=np.array(rewards)
        #Instead of computing these in surrogate as it is usually done, we do it here to save time
        returns,advantage,td_error=([list() for _ in range(len(states))] for _ in range(3))
        tmp_r = np.zeros((20,1))
        returns = [0 for i in range(len(v))]
        advantage = [0 for i in range(len(v))]
        tmp_adv = np.zeros((20, 1))
        next_value=np.zeros((20,1))
        for i in reversed(range(len(states))):
            if i<len(states)-1: 
                next_value = v[i+1]
            tmp_r = rewards[i] + discount  * tmp_r
            tmp_td_error = rewards[i] + discount * next_value - v[i]
            tmp_adv = tmp_adv * eta * discount  + tmp_td_error
            returns[i] = tmp_r
            advantage[i] = tmp_adv
        returns=np.array(returns)
        advantage=np.array(advantage)
        #NORMALIZING:
#         mean = np.mean(returns, axis=1) 
#         std = np.std(returns, axis=1) + 1.0e-10
#         returns = (returns - mean[:,np.newaxis])/std[:,np.newaxis]
        #notice we only normalize advantage not rewards which are sparse and low (esp. at beginning)
        returns = torch.tensor(returns, dtype=torch.float, device=device)
        mean = np.mean(advantage, axis=1) 
        std = np.std(advantage, axis=1) + 1.0e-10
        advantage = (advantage - mean[:,np.newaxis])/std[:,np.newaxis]
        advantage = torch.tensor(advantage, dtype=torch.float, device=device)
        for _ in range(SGD_epoch):
            L = -surrogate(agent.policy, torch.tensor(old_log_probs).to(device), states, actions,returns,advantage,
                                  c1=c1, epsilon=epsilon,delta=delta,
                                  beta1=beta,beta2=beta,loss_kind=loss_kind) #negative sign is important as maximizing is the goal
            agent.optimizer.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(),clip) #clipping the gradients
            agent.optimizer.step()
            del L
        #decaying the clipping and scalar factor hyperparams 
        epsilon*=.999
        beta*=.995
        mean_rewards.append(np.mean(total_rewards))
        min_rewards.append(np.min(total_rewards))
        max_rewards.append(np.max(total_rewards))
        mavg_rewards.append(np.mean(mean_rewards[-100:]))
        if e%10==0:
            print("Episode: {}, score: {}, max:{}, min: {}, moving avg: {}".format(e,np.mean(total_rewards),np.max(total_rewards),
                                                                   np.min(total_rewards),np.mean(mean_rewards[-100:])))
        if np.mean(mean_rewards[-100:])>target_score:
            print("Environment Solved in {} Epsiodes".format(e))
            stats+=f'solved{e}'
            torch.save(agent.policy.state_dict(),stats+f'.pth')
            np.savetxt(stats+f'mean', mean_rewards)
            np.savetxt(stats+f'min', min_rewards)
            np.savetxt(stats+f'max', max_rewards)
            np.savetxt(stats+f'mavg', mavg_rewards)
            break
        timer.update(e)    
    timer.finish()
    
    plt.plot(mean_rewards)
    return mean_rewards,min_rewards,max_rewards,mavg_rewards,stats