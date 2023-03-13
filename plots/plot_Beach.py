import itertools
import pickle
import string

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from cycler import cycler

from experiments import args_parser
from mpl_toolkits.mplot3d import Axes3D


def plot():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 24,
        "font.sans-serif": ["Helvetica"],
    })

    i = 1
    game = 'Beach-Graphon'

    """ Exploitability plot """
    clist = itertools.cycle(cycler(color='rbgcmyk'))
    linestyle_cycler = itertools.cycle(cycler('linestyle',['-','--',':','-.']))
    subplot = plt.subplot(1, 3, i)
    subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=subplot.transAxes,
            weight='bold')
    i += 1

    for graphon in ['power']:
        for solver in ['omd']:
            graphon_label = r'$W_\mathrm{power}$' if graphon=='power' else r'$W_\mathrm{cutoff}$'

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
                'num_alphas': 10 if game=='Beach-Graphon' else 50,
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

    """ 2D heatmap final timestep """
    subplot = plt.subplot(1, 3, i)
    subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=subplot.transAxes,
            weight='bold')
    i += 1
    for graphon in ['power']:
        for solver in ['omd']:
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
                'num_alphas': 10 if game=='Beach-Graphon' else 50,
                'alpha': 0.5,
                'env_params': 0,
            })
            with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
                result = pickle.load(f)

            heatmap = np.zeros(10 * 10)
            num_alphas = 10
            alphas = np.linspace(1 / num_alphas / 2, 1 - 1 / num_alphas / 2, num_alphas)
            for idx, (alpha, x) in zip(range(10 * 10), itertools.product(range(len(alphas)), range(10))):
                heatmap[idx] = result[-1]['simulator']['mus'][-1][alpha][x]

            """ Plot """
            im = plt.imshow(heatmap.reshape((10, 10)), interpolation='none',
                       extent=[0, 10, 0, 1], aspect=10, cmap='hot', origin='lower', vmin=0)
            plt.xticks([0, 5, 10])
            plt.yticks(alphas[[0,3,6,9]])
            plt.xlabel(r'State $x$')
            plt.ylabel(r'Graphon index $\alpha$')

            cb1 = plt.colorbar(im,fraction=0.046, pad=0.04)
            # cb1.set_ticks([0, 0.5, 1], update_ticks=True)
            cb1.set_label(r'Final mean-field $\mu^\alpha_{49}(x)$')

    """ 3D Surface Plot """
    subplot = plt.subplot(1, 3, i, projection='3d')
    subplot.text2D(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=subplot.transAxes,
             weight='bold')
    i += 1
    for graphon in ['power']:
        for solver in ['omd']:
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
                'num_alphas': 10 if game=='Beach-Graphon' else 50,
                'alpha': 0.5,
                'env_params': 0,
            })
            with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
                result = pickle.load(f)

            X, Y = np.meshgrid(list(range(50)), list(range(10)))
            means = np.array(result[-1]['simulator']['mus'])
            masses = [means[np.ravel(X),alpha_idx,np.ravel(Y)] for alpha_idx in range(10)]
            zs = np.mean(masses, axis=0)
            Z = zs.reshape(X.shape)
            sp = subplot.plot_surface(X, Y, Z, cmap=pl.cm.plasma_r, vmin=0)

            subplot.set_xlabel(r'$t$')
            subplot.set_ylabel(r'$x$')
            subplot.set_zlabel(r'$\int \mu^\alpha_t(x) \, \mathrm d\alpha$')
            subplot.set_yticks([0,5,10])

            cb1 = plt.colorbar(sp, fraction=0.04, pad=0.13)
            # cb1.set_ticks([0, 0.5, 1], update_ticks=True)
            # cb1.set_label(r'Mean field $\mu_t(x)$')

    """ Finalize plot """
    plt.gcf().set_size_inches(20, 6)
    plt.tight_layout(w_pad=-0.1)
    plt.savefig('./figures/exploitability.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.savefig('./figures/exploitability.png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
