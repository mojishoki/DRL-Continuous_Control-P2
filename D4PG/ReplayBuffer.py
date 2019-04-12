import random
import numpy as np
from collections import namedtuple, deque, OrderedDict 
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, rollout_length):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            rollout_length (int): length of each trajectory used for training
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.rollout_length=rollout_length
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    
    def sample(self):
        """Randomly sample a batch of experiences of trajectories with length `rollout_length` from memory."""
        starts = random.sample(list(range(len(self.memory)-self.rollout_length)), k=self.batch_size)
        states,actions,rewards,next_states,dones=(list() for _ in range(5))
        for i in range(self.rollout_length):
            experiences= [self.memory[start+i] for start in starts]
            states.append(torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device))
            actions.append(torch.from_numpy(np.vstack([e.action for e in experiences  if e is not None])).float().to(device))
            rewards.append(torch.from_numpy(np.vstack([e.reward for e in experiences  if e is not None])).float().to(device))
            next_states.append(torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device))
            dones.append(torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device))

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)