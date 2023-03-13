import numpy as np

from games.graphon_mfg import FiniteGraphonMeanFieldGame
from simulator.mean_fields.base import MeanField
from solver.base import Solver
from solver.policy.finite_policy import QMaxPolicy, QSoftMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


class DiscretizedGraphonExactOMDSolverFinite(Solver):
    """
    Exact OMD solutions for finite state spaces
    """

    def __init__(self, eta=1, num_alphas=100, **kwargs):
        super().__init__(**kwargs)
        self.num_alphas = num_alphas
        self.alphas = np.linspace(1 / num_alphas / 2, 1 - 1 / num_alphas / 2, num_alphas)
        self.y = None
        self.omd_coeff = 0.1
        self.eta = eta

    def solve(self, mfg: FiniteGraphonMeanFieldGame, mu: MeanField, **kwargs):
        Q_alphas = []

        for alpha in self.alphas:
            Vs = []
            Qs = []
            curr_V = [0 for _ in range(mfg.agent_observation_space[1].n)]

            for t in range(mfg.time_steps).__reversed__():
                Q_t = []
                for x in range(mfg.agent_observation_space[1].n):
                    x = tuple([alpha, x])
                    Q_tx = np.array([mfg.reward(t, x, u, mu) + (1 - mfg.done(t, x, u, mu)) *
                                     np.vdot(curr_V, mfg.transition_probs(t, x, u, mu))
                                     for u in range(mfg.agent_action_space.n)])
                    Q_t.append(Q_tx)
                curr_V = [np.max(Q_t[x]) for x in range(len(curr_V))]

                Vs.append(curr_V)
                Qs.append(Q_t)

            Vs.reverse()
            Qs.reverse()
            Q_alphas.append(Qs)

        if self.y is None:
            self.y = self.omd_coeff * np.array(Q_alphas)
        else:
            self.y = self.y + self.omd_coeff * np.array(Q_alphas)

        policy = DiscretizedGraphonFeedbackPolicy(mfg.agent_observation_space, mfg.agent_action_space,
                                                  [
                                                      QSoftMaxPolicy(mfg.agent_observation_space, mfg.agent_action_space, Qs, 1 / self.eta)
                                                      for Qs, alpha in zip(self.y, self.alphas)
                                                  ], self.alphas)

        return policy, {"Q": self.y}
