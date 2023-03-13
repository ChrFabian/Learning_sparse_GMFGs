import itertools
import pickle
import string

import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Discrete
from matplotlib import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable

from experiments import args_parser
from solver.policy.finite_policy import QSoftMaxPolicy, QMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


def plot():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 24,
        # "axes.linewidth": 3,
    })
    cmap = pl.cm.plasma_r
    cmap_alt = pl.cm.binary

    num_alpha = 25
    game = 'Cyber-Graphon'
    graphon = 'power'
    solver = 'omd'
    alpha_idxs = range(0, num_alpha, 1)
    colors = cmap(np.linspace(0, 1, num_alpha))
    graphon_label = graphon
    graphon_label += ' graphon'
    clist = itertools.cycle(cycler(color='rbgcmyk'))
    linestyle_cycler = itertools.cycle(cycler('linestyle',['-','--',':','-.']))

    i = 1
    plt.subplot(2, 3, i)
    plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=plt.gca().transAxes,
        size=22, weight='bold')
    i += 1

    args = args_parser.generate_config_from_kw(**{
        'game': game,
        'graphon': graphon,
        'solver': solver,
        'simulator': 'exact',
        'evaluator': 'exact',
        'eval_solver': 'exact',
        'iterations': 250,
        'total_iterations': 500,
        'eta': 1 if solver == 'omd' else 0,
        'results_dir': None,
        'exp_name': None,
        'verbose': 0,
        'num_alphas': num_alpha,
        'alpha': 0.5,
        'env_params': 0,
    })
    with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
        result = pickle.load(f)
        eps = [result[t]['eval_opt']['eval_mean_returns']
                                      - result[t]['eval_pi']['eval_mean_returns']
                                      for t in range(len(result))]
    label = graphon_label
    color = clist.__next__()['color']
    linestyle = linestyle_cycler.__next__()['linestyle']

    plt.semilogy(range(len(eps)), eps, linestyle, color=color, label=label, alpha=0.5)

    # plt.legend()
    plt.grid('on')
    plt.xlabel(r'Iteration $n$', fontsize=22)
    plt.ylabel(r'Exploitability $\Delta J$', fontsize=22)
    plt.xlim([0, len(eps) - 1])
    plt.ylim([10 ** -2, 10 ** 1])

    for plot_num in range(5):
        plt.subplot(2, 3, i)
        plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=plt.gca().transAxes,
            size=22, weight='bold')
        i += 1

        args = args_parser.generate_config_from_kw(**{
            'game': game,
            'graphon': graphon,
            'solver': solver,
            'simulator': 'exact',
            'evaluator': 'exact',
            'eval_solver': 'exact',
            'iterations': 250,
            'total_iterations': 500,
            'eta': 1 if solver == 'omd' else 0,
            'results_dir': None,
            'exp_name': None,
            'verbose': 0,
            'num_alphas': num_alpha,
            'alpha': 0.5,
            'env_params': 0,
        })
        with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
            result = pickle.load(f)
            last_means = None
            if plot_num < 4:
                """ Reconstruct policy"""
                Q_alphas = result[-1]['solver']['Q']
                alphas = np.linspace(0, 1, num_alpha)
                policy = DiscretizedGraphonFeedbackPolicy(Discrete(2),
                                                          Discrete(2),
                                                          [
                                                              QSoftMaxPolicy(Discrete(2),
                                                                             Discrete(2),
                                                                             Qs,
                                                                             1 / 1)
                                                              for Qs, alpha in zip(Q_alphas, alphas)
                                                          ], alphas) if solver=='omd' else \
                        DiscretizedGraphonFeedbackPolicy(Discrete(2),
                                                         Discrete(2),
                                                         [
                                                             QMaxPolicy(Discrete(2),
                                                                            Discrete(2),
                                                                            Qs)
                                                             for Qs, alpha in zip(Q_alphas, alphas)
                                                         ], alphas)

                # plotted_alphas = np.linspace(1 / num_alpha / 2, 1 - 1 / num_alpha / 2, num_alpha)
                plotted_alphas = np.linspace(0, 1, 200)
                for alpha_idx in range(len(plotted_alphas)):
                    means = []
                    for t in range(50):
                        means.append(policy.pmf(t, tuple([plotted_alphas[alpha_idx], plot_num]))[1])

                    for t in range(50):
                        color = cmap_alt(means[t])
                        plt.plot([t, t + 1], 2 * [plotted_alphas[alpha_idx]], color=color, label='_nolabel_',
                                 linewidth=2)

            else:
                plotted_alphas = np.linspace(1 / num_alpha / 2, 1 - 1 / num_alpha / 2, num_alpha)
                for alpha_idx in range(len(plotted_alphas)):
                    means = []
                    for t in range(50):
                        mean = np.array(result[-1]['simulator']['mus'][t][alpha_idx][2]) \
                               + result[-1]['simulator']['mus'][t][alpha_idx][0]
                        means.append(mean)

                    color = colors[alpha_idx]

                    plt.plot(range(50), means, color=color, label='_nolabel_', linewidth=2)
                    # if last_means is not None:
                    #     plt.fill_between(range(50), means, last_means, color=color)
                    last_means = means

        # plt.legend()
        plt.grid('on')
        if plot_num <= 3:
            plt.xlabel(fr'Time $t$')
            plt.ylabel(fr'Graphon index $\alpha$')
            plt.ylim([0, 1])
            plt.xlim([0, 49])

            divider = make_axes_locatable(plt.gca())
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap_alt, orientation='vertical')
            cb1.set_ticks([0, 1], update_ticks=True)
            plt.gcf().add_axes(ax_cb)
            state_label = ['DI', 'DS', 'UI', 'US']
            cb1.set_label(fr'$\pi^\alpha_t(1 \mid {state_label[plot_num]})$')
        else:
            plt.xlabel(fr'Time $t$')
            plt.ylabel(fr'Infection prob. $\mu^\alpha_t(I)$')
            # plt.ylim([0, 1])
            plt.xlim([0, 49])

            divider = make_axes_locatable(plt.gca())
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
            cb1.set_ticks([0, 1], update_ticks=True)
            plt.gcf().add_axes(ax_cb)
            plt.title(r'$\alpha$')

    plt.gcf().set_size_inches(18, 8)
    plt.tight_layout()
    plt.savefig('./figures/Cyber_heatmap.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.savefig('./figures/Cyber_heatmap.png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
