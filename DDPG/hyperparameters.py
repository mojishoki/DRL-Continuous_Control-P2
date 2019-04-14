#FOR DDPG
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # Weight_decay for Adam optimizer
UPDATE_EVERY = 20       # every update_every steps update
LEARN_NUM = 10          # learn_num times
SEED=1                  # random seed