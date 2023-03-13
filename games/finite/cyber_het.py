import numpy as np
import torch
from gym.spaces import Discrete
from torch.distributions import Categorical

from games.graphon_mfg import FiniteGraphonMeanFieldGame


class HeterogeneousCyberGraphon(FiniteGraphonMeanFieldGame):
    """
    Models the heterogeneous Cybersecurity game.
    """

    def __init__(self, graphon=(lambda x, y: 1-max(x, y)), time_steps: int = 50, mu_0=(0.125,) * 8,
                 q_rec_D: float = 0.4, q_rec_U: float = 0.3, lambda_wait: float = 0.3, v_H: float = 0.1,
                 z_inf_D: float = 0.05, z_inf_U: float = 0.1, k_D: float = 0.7, k_I: float = 2.0,
                 beta_DD: float = 0.1, beta_UD: float = 0.2, beta_DU: float = 0.7, beta_UU: float = 0.8,
                 beta_DPriD: float = 0.2, beta_UPriD: float = 0.3, beta_DPriU: float = 0.9, beta_UPriU: float = 1.0,
                 z_inf_D_Pri: float = 0.05, z_inf_U_Pri: float = 0.1, k_D_Pri: float = 0.6, k_I_Pri: float = 2.0,
                 q_rec_D_Pri: float = 0.4, q_rec_U_Pri: float = 0.3, **kwargs):
        self.graphon = graphon
        self.q_rec_D = q_rec_D
        self.q_rec_U = q_rec_U
        self.q_rec_D_Pri = q_rec_D_Pri
        self.q_rec_U_Pri = q_rec_U_Pri
        self.lambda_wait = lambda_wait
        self.lambda_wait_Pri = lambda_wait
        self.v_H = v_H
        self.z_inf_D = z_inf_D
        self.z_inf_U = z_inf_U
        self.z_inf_D_Pri = z_inf_D_Pri
        self.z_inf_U_Pri = z_inf_U_Pri
        self.beta_DD = beta_DD
        self.beta_UD = beta_UD
        self.beta_DU = beta_DU
        self.beta_UU = beta_UU
        self.beta_DPriD = beta_DPriD
        self.beta_UPriD = beta_UPriD
        self.beta_DPriU = beta_DPriU
        self.beta_UPriU = beta_UPriU
        self.k_D = k_D
        self.k_I = k_I
        self.k_D_Pri = k_D_Pri
        self.k_I_Pri = k_I_Pri

        # States: PriDI PriDS PriUI PriUS CorDI CorDS CorUI CorUS
        def initial_state_distribution(x):
            return Categorical(probs=torch.tensor(mu_0))
        agent_observation_space = Discrete(8)
        agent_action_space = Discrete(2)
        super().__init__(agent_observation_space, agent_action_space, time_steps, initial_state_distribution, graphon)

    def transition_probs_g_fast(self, t, x, u, g):
        g_DI = min(1., g.evaluate_integral(t, lambda dy: (dy[:, 1] == 0) + (dy[:, 1] == 4)))
        g_UI = min(1., g.evaluate_integral(t, lambda dy: (dy[:, 1] == 2) + (dy[:, 1] == 6)))
        q_inf_D_Pri = np.minimum(1, self.v_H * self.z_inf_D_Pri \
                      + self.beta_DPriD * g_DI \
                      + self.beta_UPriD * g_UI \
                      - self.v_H * self.z_inf_D_Pri * self.beta_DPriD * g_DI \
                      - self.v_H * self.z_inf_D_Pri * self.beta_UPriD * g_UI \
                      - self.beta_DPriD * self.beta_UPriD * g_DI * g_UI \
                      + self.v_H * self.z_inf_D_Pri * self.beta_DPriD * self.beta_UPriD * g_DI * g_UI)
        q_inf_U_Pri = np.minimum(1, self.v_H * self.z_inf_U_Pri \
                      + self.beta_DPriU * g_DI \
                      + self.beta_UPriU * g_UI \
                      - self.v_H * self.z_inf_U_Pri * self.beta_DPriU * g_DI \
                      - self.v_H * self.z_inf_U_Pri * self.beta_UPriU * g_UI \
                      - self.beta_DPriU * self.beta_UPriU * g_DI * g_UI \
                      + self.v_H * self.z_inf_U_Pri * self.beta_DPriU * self.beta_UPriU * g_DI * g_UI)
        q_inf_D = np.minimum(1, self.v_H * self.z_inf_D \
                      + self.beta_DD * g_DI \
                      + self.beta_UD * g_UI \
                      - self.v_H * self.z_inf_D * self.beta_DD * g_DI \
                      - self.v_H * self.z_inf_D * self.beta_UD * g_UI \
                      - self.beta_DD * self.beta_UD * g_DI * g_UI \
                      + self.v_H * self.z_inf_D * self.beta_DD * self.beta_UD * g_DI * g_UI)
        q_inf_U = np.minimum(1, self.v_H * self.z_inf_U \
                      + self.beta_DU * g_DI \
                      + self.beta_UU * g_UI \
                      - self.v_H * self.z_inf_U * self.beta_DU * g_DI \
                      - self.v_H * self.z_inf_U * self.beta_UU * g_UI \
                      - self.beta_DU * self.beta_UU * g_DI * g_UI \
                      + self.v_H * self.z_inf_U * self.beta_DU * self.beta_UU * g_DI * g_UI)
        q_rec_D_Pri = self.q_rec_D_Pri
        q_rec_U_Pri = self.q_rec_U_Pri
        q_rec_D = self.q_rec_D
        q_rec_U = self.q_rec_U

        transition_matrix = np.array([
            [(1 - u * self.lambda_wait_Pri) * (1 - q_rec_D_Pri), (1 - u * self.lambda_wait_Pri) * (q_rec_D_Pri), (u * self.lambda_wait_Pri) * (1 - q_rec_D_Pri), (u * self.lambda_wait_Pri) * (q_rec_D_Pri), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)],
            [(1 - u * self.lambda_wait_Pri) * (q_inf_D_Pri), (1 - u * self.lambda_wait_Pri) * (1 - q_inf_D_Pri), (u * self.lambda_wait_Pri) * (q_inf_D_Pri), (u * self.lambda_wait_Pri) * (1 - q_inf_D_Pri), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)],
            [(u * self.lambda_wait_Pri) * (1 - q_rec_U_Pri), (u * self.lambda_wait_Pri) * (q_rec_U_Pri), (1 - u * self.lambda_wait_Pri) * (1 - q_rec_U_Pri), (1 - u * self.lambda_wait_Pri) * (q_rec_U_Pri), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)],
            [(u * self.lambda_wait_Pri) * (q_inf_U_Pri), (u * self.lambda_wait_Pri) * (1 - q_inf_U_Pri), (1 - u * self.lambda_wait_Pri) * (q_inf_U_Pri), (1 - u * self.lambda_wait_Pri) * (1 - q_inf_U_Pri), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)],
            [np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), (1 - u * self.lambda_wait) * (1 - q_rec_D), (1 - u * self.lambda_wait) * (q_rec_D), (u * self.lambda_wait) * (1 - q_rec_D), (u * self.lambda_wait) * (q_rec_D)],
            [np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), (1 - u * self.lambda_wait) * (q_inf_D), (1 - u * self.lambda_wait) * (1 - q_inf_D), (u * self.lambda_wait) * (q_inf_D), (u * self.lambda_wait) * (1 - q_inf_D)],
            [np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), (u * self.lambda_wait) * (1 - q_rec_U), (u * self.lambda_wait) * (q_rec_U), (1 - u * self.lambda_wait) * (1 - q_rec_U), (1 - u * self.lambda_wait) * (q_rec_U)],
            [np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), (u * self.lambda_wait) * (q_inf_U), (u * self.lambda_wait) * (1 - q_inf_U), (1 - u * self.lambda_wait) * (q_inf_U), (1 - u * self.lambda_wait) * (1 - q_inf_U)],
        ])

        return transition_matrix[x[:, 1].astype(int), ..., np.array(range(len(x))) ]

    def transition_probs_g(self, t, x, u, g):
        g_DI = min(1., g.evaluate_integral(t, lambda dy: (dy[1] == 0) + (dy[1] == 4)))
        g_UI = min(1., g.evaluate_integral(t, lambda dy: (dy[1] == 2) + (dy[1] == 6)))
        q_inf_D_Pri = min(1., self.v_H * self.z_inf_D_Pri \
                      + self.beta_DPriD * g_DI \
                      + self.beta_UPriD * g_UI \
                      - self.v_H * self.z_inf_D_Pri * self.beta_DPriD * g_DI \
                      - self.v_H * self.z_inf_D_Pri * self.beta_UPriD * g_UI \
                      - self.beta_DPriD * self.beta_UPriD * g_DI * g_UI \
                      + self.v_H * self.z_inf_D_Pri * self.beta_DPriD * self.beta_UPriD * g_DI * g_UI)
        q_inf_U_Pri = min(1., self.v_H * self.z_inf_U_Pri \
                      + self.beta_DPriU * g_DI \
                      + self.beta_UPriU * g_UI \
                      - self.v_H * self.z_inf_U_Pri * self.beta_DPriU * g_DI \
                      - self.v_H * self.z_inf_U_Pri * self.beta_UPriU * g_UI \
                      - self.beta_DPriU * self.beta_UPriU * g_DI * g_UI \
                      + self.v_H * self.z_inf_U_Pri * self.beta_DPriU * self.beta_UPriU * g_DI * g_UI)
        q_inf_D = min(1., self.v_H * self.z_inf_D \
                      + self.beta_DD * g_DI \
                      + self.beta_UD * g_UI \
                      - self.v_H * self.z_inf_D * self.beta_DD * g_DI \
                      - self.v_H * self.z_inf_D * self.beta_UD * g_UI \
                      - self.beta_DD * self.beta_UD * g_DI * g_UI \
                      + self.v_H * self.z_inf_D * self.beta_DD * self.beta_UD * g_DI * g_UI)
        q_inf_U = min(1., self.v_H * self.z_inf_U \
                      + self.beta_DU * g_DI \
                      + self.beta_UU * g_UI \
                      - self.v_H * self.z_inf_U * self.beta_DU * g_DI \
                      - self.v_H * self.z_inf_U * self.beta_UU * g_UI \
                      - self.beta_DU * self.beta_UU * g_DI * g_UI \
                      + self.v_H * self.z_inf_U * self.beta_DU * self.beta_UU * g_DI * g_UI)
        q_rec_D_Pri = self.q_rec_D_Pri
        q_rec_U_Pri = self.q_rec_U_Pri
        q_rec_D = self.q_rec_D
        q_rec_U = self.q_rec_U

        transition_matrix = np.array([
            [(1 - u * self.lambda_wait_Pri) * (1 - q_rec_D_Pri), (1 - u * self.lambda_wait_Pri) * (q_rec_D_Pri), (u * self.lambda_wait_Pri) * (1 - q_rec_D_Pri), (u * self.lambda_wait_Pri) * (q_rec_D_Pri), 0, 0, 0, 0],
            [(1 - u * self.lambda_wait_Pri) * (q_inf_D_Pri), (1 - u * self.lambda_wait_Pri) * (1 - q_inf_D_Pri), (u * self.lambda_wait_Pri) * (q_inf_D_Pri), (u * self.lambda_wait_Pri) * (1 - q_inf_D_Pri), 0, 0, 0, 0],
            [(u * self.lambda_wait_Pri) * (1 - q_rec_U_Pri), (u * self.lambda_wait_Pri) * (q_rec_U_Pri), (1 - u * self.lambda_wait_Pri) * (1 - q_rec_U_Pri), (1 - u * self.lambda_wait_Pri) * (q_rec_U_Pri), 0, 0, 0, 0],
            [(u * self.lambda_wait_Pri) * (q_inf_U_Pri), (u * self.lambda_wait_Pri) * (1 - q_inf_U_Pri), (1 - u * self.lambda_wait_Pri) * (q_inf_U_Pri), (1 - u * self.lambda_wait_Pri) * (1 - q_inf_U_Pri), 0, 0, 0, 0],
            [0, 0, 0, 0, (1 - u * self.lambda_wait) * (1 - q_rec_D), (1 - u * self.lambda_wait) * (q_rec_D), (u * self.lambda_wait) * (1 - q_rec_D), (u * self.lambda_wait) * (q_rec_D)],
            [0, 0, 0, 0, (1 - u * self.lambda_wait) * (q_inf_D), (1 - u * self.lambda_wait) * (1 - q_inf_D), (u * self.lambda_wait) * (q_inf_D), (u * self.lambda_wait) * (1 - q_inf_D)],
            [0, 0, 0, 0, (u * self.lambda_wait) * (1 - q_rec_U), (u * self.lambda_wait) * (q_rec_U), (1 - u * self.lambda_wait) * (1 - q_rec_U), (1 - u * self.lambda_wait) * (q_rec_U)],
            [0, 0, 0, 0, (u * self.lambda_wait) * (q_inf_U), (u * self.lambda_wait) * (1 - q_inf_U), (1 - u * self.lambda_wait) * (q_inf_U), (1 - u * self.lambda_wait) * (1 - q_inf_U)],
        ])

        return transition_matrix[x[1]]

    def reward_g(self, t, x, u, g):
        return - self.k_D * ((x[1] == 0) + (x[1] == 1)) - self.k_I * ((x[1] == 0) + (x[1] == 2)) \
               - self.k_D_Pri * ((x[1] == 4) + (x[1] == 5)) - self.k_I_Pri * ((x[1] == 4) + (x[1] == 6))
