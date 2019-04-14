#For D4PG
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4        # learning rate of the actor, this time for D4PG we choose different lr for actor and critic 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 1e-5    #weight decay parameter for adam optimizer
UPDATE_EVERY = 20
LEARN_NUM = 10
EPSILON = 1.0           # explore->exploit noise process added to act step for gaussian noise, works if eps_gauss=True in Agent
EPSILON_DECAY = 1e-6   #for linear rate of decay  
ATOMS=51            #number of atoms of categorical distribution
ROLLOUT_LENGTH=5    #rollout length taken for trajectory 
V_MAX=4             #V_MAX and V_MIN are gonna be the interval within which atoms reside
V_MIN=0
HARD_UPDATE=350    #hard update of target networks at each hard_update many steps 
SEED=1
