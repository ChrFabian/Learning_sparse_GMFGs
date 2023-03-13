import itertools

import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import string

from experiments import args_parser
from solver.policy.finite_policy import QSoftMaxPolicy, QMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy
from gym.spaces import Discrete
from mpl_toolkits.axes_grid1 import make_axes_locatable


def run_once(simulator, fixed_alphas):
    done = 0
    simulator.reset(fixed_alphas)
    alphas = [simulator.x[i][0] for i in range(len(simulator.x))]
    returns = np.zeros_like(alphas)
    while not done:
        _, rewards, done, _ = simulator.step()
        returns += np.array(rewards)
    return alphas, returns


def plot():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 22,
        # "axes.linewidth": 3,
    })
    cmap = pl.cm.plasma_r

    # num_players_points = range(2, 200, 1)
    # num_betas = 50
    num_players_points = range(2, 100, 1)
    num_betas = 50
    num_run_ids = range(0, 1)
    betas = np.linspace(1 / num_betas / 2, 1 - 1 / num_betas / 2, num_betas)

    games = ['Cyber-Graphon', 'Beach-Graphon']#, 'Cyber-Het-Graphon', 'Systemic-Risk-Graphon']
    graphon = ('power')

    """ Second half of plot """
    for game in games:
        plt.figure()
        plt.plot()
        # plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=plt.gca().transAxes,
        #         size=22, weight='bold')

        solver = 'omd'
        eta = 1
        num_alphas = 10 if game == "Beach-Graphon" else 25
        graphon_label = r'$W_\mathrm{power}$'

        args = args_parser.generate_config_from_kw(**{
            'game': game,
            'graphon': graphon,
            'solver': solver,
            'simulator': 'exact',
            'evaluator': 'exact',
            'eval_solver': 'exact',
            'iterations': 50,
            'total_iterations': 500,
            'eta': eta,
            'results_dir': None,
            'exp_name': None,
            'verbose': 0,
            'num_alphas': num_alphas,
            'alpha': 0.5,
            'env_params': 0,
        })

        with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
            result = pickle.load(f)

        """ Reconstruct policy """
        mfg = args["game"](**args["game_config"])
        Q_alphas = result[-1]['solver']['Q']
        alphas = np.linspace(1 / num_alphas / 2, 1 - 1 / num_alphas / 2, num_alphas)
        policy = DiscretizedGraphonFeedbackPolicy(mfg.agent_observation_space[1],
                                                  mfg.agent_action_space,
                                                  [
                                                      QSoftMaxPolicy(mfg.agent_observation_space[1],
                                                                     mfg.agent_action_space,
                                                                     Qs,
                                                                     1 / eta) if eta > 0 else
                                                      QMaxPolicy(mfg.agent_observation_space[1],
                                                                 mfg.agent_action_space,
                                                                 Qs)
                                                      for Qs, alpha in zip(Q_alphas, alphas)
                                                  ], alphas)

        mus = np.array(result[-1]['simulator']['mus'])
        mean_states_mu = np.mean(mus, axis=1)

        heatmap = np.zeros(len(num_players_points) * num_betas)
        for run_id in num_run_ids:
            for idx, (beta, num_players) in zip(range(num_betas * len(num_players_points)), itertools.product(betas, num_players_points)):
                try:
                    with open(args['experiment_directory'] + f'nagent_seeded_{run_id}_{num_players}_{beta:.2f}.pkl', 'rb') as f:
                        returns_and_alphas = pickle.load(f)

                    state_trajectories = [returns_and_alphas[i][2] for i in range(len(returns_and_alphas))]
                    state_trajectories_no_alpha = [[[state_trajectories[i][j][k][0][1]
                                                     for k in range(len(state_trajectories[i][j]))]
                                                    for j in range(len(state_trajectories[i]))]
                                                   for i in range(len(state_trajectories))]
                    state_traj_array = np.array(state_trajectories_no_alpha)
                    diffs_to_mu = np.sum([np.sum(np.abs(np.mean(state_traj_array == x, axis=-1) - mean_states_mu[:,x]), axis=-1) for x in range(mfg.agent_observation_space[1].n)], axis=0)
                    mean = np.mean(diffs_to_mu)
                    heatmap[idx] = mean
                except FileNotFoundError:
                    print("Not Found!")
                    pass

        """ Plot """
        plt.imshow(heatmap.reshape((num_betas, len(num_players_points))),  # vmin=0,
                        extent=[2, 2+len(num_players_points), 0, 1], aspect=num_betas, cmap='hot', interpolation='nearest', origin='lower')
        # plt.xticks([0, 0.5, 1])
        # plt.yticks([0, 0.5, 1])
        plt.xlabel(r'Number of agents $N$')
        plt.ylabel(r'Sparsity parameter $\beta$')

        cb1 = plt.colorbar()
        # cb1.set_ticks([0, 0.5, 1], update_ticks=True)
        cb1.set_label(r'Deviation $\Delta \mu$')

        plt.gcf().set_size_inches(14, 5)
        plt.tight_layout()
        plt.savefig(f'./figures/heatmap_{game}_finite_comparison.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()


if __name__ == '__main__':
    plot()
