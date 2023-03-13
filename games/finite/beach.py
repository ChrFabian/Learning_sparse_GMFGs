import numpy as np
import torch
from gym.spaces import Discrete
from torch.distributions import Categorical

from games.graphon_mfg import FiniteGraphonMeanFieldGame


class BeachGraphon(FiniteGraphonMeanFieldGame):
    """
    Models the Beach Bar Process.
    """

    def __init__(self, graphon=(lambda x, y: 1-max(x, y)), time_steps: int = 50, N_states=10,
                 noise_prob: float = 0.1, **kwargs):
        self.graphon = graphon
        self.noise_prob = noise_prob
        self.N_states = N_states

        # States: 0 1 2 3 4 Bar 6 7 8 9
        # Actions: Left Stay Right
        def initial_state_distribution(x):
            return Categorical(probs=torch.tensor([1/N_states] * N_states))
        agent_observation_space = Discrete(N_states)
        agent_action_space = Discrete(3)
        super().__init__(agent_observation_space, agent_action_space, time_steps, initial_state_distribution, graphon)

    def transition_probs_g(self, t, x, u, g):
        transition_probs = np.zeros(self.N_states)
        transition_probs[(x[1]-1 + (u-1)) % 10] = self.noise_prob/2
        transition_probs[(x[1] + (u-1)) % 10] = (1-self.noise_prob)
        transition_probs[(x[1]+1 + (u-1)) % 10] = self.noise_prob/2
        return transition_probs

    def reward_g(self, t, x, u, g):
        r_x = -abs(x[1] - self.N_states/2) / (self.N_states/2)
        r_a = -((u==0) + (u==2)) / (self.N_states/2)
        r_mu = -g.evaluate_integral(t, lambda dy: dy[1] == x[1]) * 3
        return r_x + r_a + r_mu
