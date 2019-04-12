import numpy as np
import matplotlib.pyplot as plt

def agent_play(agent,env,brain_name,train_mode=False,view_episodes=1,ppo=False):
    """
    Watch the agent play in Unity: 
            train_mode: if you want to watch the agent play, set False, if you want to run many plays in short time, set True
            Returns score_list            
    """
    score_list = []   
    for i in range(1,view_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]# reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(len(env_info.agents))  
        while True: 
            if ppo:
                actions = agent.act(states)[0]
            else:
                actions = agent.act(states,watch_phase=True)   # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states
            if np.any(dones):                                  # exit loop if episode finished
                break
        score_list.append(np.mean(scores))
        print('epsiode {} max {:.1f} min {:.1f} average {:.1f} '.format(i,np.max(scores),np.min(scores),np.mean(scores)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(score_list)), score_list)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    return score_list