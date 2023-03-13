import itertools
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import string

from matplotlib import cycler

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

    num_players_points = range(2, 100, 1)
    num_run_ids = range(0, 1)

    i = 1
    games = ['Cyber-Graphon', 'Beach-Graphon']
    labels = ['Cyber', 'Beach']
    graphon = ('power')
    clist = itertools.cycle(cycler(color='rbgcmyk'))
    linestyle_cycler = itertools.cycle(cycler('linestyle',['-','--',':','-.']))

    plt.plot()
    for game, label in zip(games, labels):
        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
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
        # mean_infecteds_mu = mean_states_mu[:,1] + mean_states_mu[:,4] + mean_states_mu[:,5]

        for run_id in num_run_ids:
            mean_infecteds = []
            ci_infecteds = []
            for num_players in num_players_points:

                try:
                    beta = 0.51
                    with open(args['experiment_directory'] + f'nagent_seeded_{run_id}_{num_players}_{beta:.2f}.pkl', 'rb') as f:
                        returns_and_alphas = pickle.load(f)
                except FileNotFoundError:
                    pass

                state_trajectories = [returns_and_alphas[i][2] for i in range(len(returns_and_alphas))]
                state_trajectories_no_alpha = [[[state_trajectories[i][j][k][0][1]
                                                 for k in range(len(state_trajectories[i][j]))]
                                                for j in range(len(state_trajectories[i]))]
                                               for i in range(len(state_trajectories))]
                state_traj_array = np.array(state_trajectories_no_alpha)
                # awares = (state_traj_array == 1) + (state_traj_array == 4) + (state_traj_array == 5)
                # mean_infected = np.mean(np.mean(awares, axis=-1), axis=0)
                # mean_infecteds.append(np.mean(np.abs(mean_infected - mean_infecteds_mu)))
                diffs_to_mu = np.sum([np.sum(np.abs(np.mean(state_traj_array == x, axis=-1) - mean_states_mu[:,x]), axis=-1) for x in range(mfg.agent_observation_space[1].n)], axis=0)
                # diffs_to_mu = np.sum(np.mean(np.mean(awares, axis=-1) - mean_infecteds_mu), axis=0)
                mean = np.mean(diffs_to_mu)
                ci = np.std(diffs_to_mu) / np.sqrt(len(state_traj_array))
                mean_infecteds.append(mean)
                ci_infecteds.append(ci)

            mean_infecteds = np.array(mean_infecteds)
            ci_infecteds = np.array(ci_infecteds)

            plt.plot(num_players_points, mean_infecteds + ci_infecteds, linestyle, color=color, label='_nolabel_', alpha=0.5)
            plt.plot(num_players_points, mean_infecteds - ci_infecteds, linestyle, color=color, label='_nolabel_', alpha=0.5)
            plt.plot(num_players_points, mean_infecteds, linestyle, color=color, label=label, alpha=0.85)
            plt.fill_between(num_players_points, mean_infecteds - ci_infecteds, mean_infecteds + ci_infecteds, color=color, alpha=0.15)

    plt.legend()
    plt.grid('on')
    plt.xlabel(r'Number of agents $N$')
    plt.ylabel(r'Mean-field deviation $\Delta \mu$')
    plt.ylim([0, 35])
    plt.xlim([3, num_players_points[-1]])

    plt.gcf().set_size_inches(14, 5)
    plt.tight_layout(h_pad=-0.1, w_pad=-0.01)
    plt.savefig('./figures/New_finite_comparison.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
