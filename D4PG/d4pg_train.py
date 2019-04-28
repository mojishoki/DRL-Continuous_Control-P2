import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import deque 
from hyperparameters import *

def d4pg_train(agent,env,brain_name,n_episodes=500, max_t=1000, target_score=30.0, window_size=100, print_every=1, train_mode=True):
    """Deep Deterministic Policy Gradient (DDPG)
    Params
    ======
        n_episodes (int)      : maximum number of training episodes
        max_t (int)           : maximum number of timesteps per episode
        window_size (int)     : moving average is taken over window_size episodes
        target_score (float)  : min moving average score to reach to solve env
        print_every (int)     : print results print_every episode
        train_mode (bool)     : if `True` set environment to training mode, if `False` watch it play while training
    """
    stats=f'Checkpoints/'
    mean_scores,min_scores,max_scores,moving_avgs=([] for _ in range(4))
    stats+=f'{agent.batch_size}{agent.buffer_size}{agent.gamma}{agent.tau}{agent.lr_actor}{agent.lr_critic}{agent.weight_decay}'
    stats+=f'{agent.update_every}{agent.learn_num}{agent.atoms}{agent.V_max}{agent.V_min}{agent.rollout_length}'
    stats+=f'{agent.noise.sigma}{agent.noise.theta}'
    scores_window = deque(maxlen=window_size)  # mean scores from most recent episodes
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name] # reset environment
        states = env_info.vector_observations                   # get current state for each agent      
        scores = np.zeros(len(env_info.agents))                 # initialize score for each agent
        agent.reset()                                           # reset the OU noise
        start_time = time.time()
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)         # select an action
            env_info = env.step(actions)[brain_name]            # send actions to environment
            next_states = env_info.vector_observations          # get next state
            rewards = env_info.rewards                          # get reward
            dones = env_info.local_done                         # see if episode has finished
            # save experience to replay buffer, perform learning step at defined interval
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)             
            states = next_states
            scores += rewards
            if np.any(dones):                                   # exit loop when episode ends
                break

        duration = time.time() - start_time
        min_scores.append(np.min(scores))             
        max_scores.append(np.max(scores))                     
        mean_scores.append(np.mean(scores))           
        scores_window.append(mean_scores[-1])         
        moving_avgs.append(np.mean(scores_window))    # save moving average
                
        if i_episode % print_every == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}\tMov. Avg: {:.1f}'.format(\
                  i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]))
                  
        if moving_avgs[-1] >= target_score and i_episode >= window_size:
            print('\nEnvironment SOLVED in {} episodes!\tMoving Average ={:.1f} over last {} episodes'.format(\
                                    i_episode, moving_avgs[-1], window_size))            
            if train_mode:
                stats+=f'solved{i_episode}'
                torch.save(agent.actor_local.state_dict(), stats+f'actor_ckpt'+f'.pth')
                torch.save(agent.critic_local.state_dict(), stats+f'critic_ckpt'+f'.pth')
                np.savetxt(stats+f'mean',mean_scores)
                np.savetxt(stats+f'min',min_scores)
                np.savetxt(stats+f'max',max_scores)
                np.savetxt(stats+f'avg',moving_avgs)
            break
            
    return mean_scores,min_scores,max_scores,moving_avgs,stats