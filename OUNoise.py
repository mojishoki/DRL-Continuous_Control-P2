import copy
import numpy as np
import random

class OUNoise:
    """Ornstein-Uhlenbeck process (almost); Check the Wikipedia page https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process for formula."""
    
    def __init__(self, size, seed, mu=0., theta=0.04, sigma=0.02):
        """Initialize parameters and noise process. Converges to a normal distribution with mean `mu` and std `sigma**2/theta`. """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample. here dt = 1"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.gauss(0,1) for i in range(len(x))])
        self.state = x + dx
        return self.state