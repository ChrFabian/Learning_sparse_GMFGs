import pickle

import numpy as np

from experiments import args_parser
from simulator.n_player_graphon_simulator import UniformGraphonNPlayerSimulator
from solver.policy.finite_policy import QSoftMaxPolicy, QMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


def run_once(simulator, fixed_alphas):
    done = 0
    state = simulator.reset(fixed_alphas)
    alphas = [simulator.x[i][0] for i in range(len(simulator.x))]
    returns = np.zeros_like(alphas)
    states = []
    while not done:
        states.append(state)
        state, rewards, done, _ = simulator.step()
        returns += np.array(rewards)
    return alphas, returns, states


def run_config(input):
    num_players = input[0]
    game = input[1]
    graphon = input[2]
    fixed_alpha = input[3]
    id_run = input[4]
    beta = input[5]

    np.random.seed(id_run)
    print(f'Running {input}')

    solver = 'omd'  # if game=="Beach-Graphon" else 'exact'
    eta = 1  # if game=="Beach-Graphon" else 0
    num_alphas = 10 if game=="Beach-Graphon" else 25

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
    mfg = args["game"](**args["game_config"])
    with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
        result = pickle.load(f)

    """ Reconstruct policy """
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

    num_alpha_trials = 1
    num_return_trials = 100

    rho_n = np.power(num_players, -beta)
    simulator = UniformGraphonNPlayerSimulator(mfg, policy, num_players, rho_n=rho_n)

    data = []
    for _ in range(num_alpha_trials):
        """ Simulate for N agents and some sampled alphas """
        # simulator.reset()
        # fixed_alphas = [simulator.x[i][0] for i in range(len(simulator.x))]
        # if fixed_alpha:
        #     fixed_alphas = np.linspace(1/num_players, 1, num_players)
        fixed_alphas = None

        for _ in range(num_return_trials):
            alphas, returns, states = run_once(simulator, fixed_alphas)
            data.append((returns, alphas, states))

        mf_returns_of_each_agent = []
        for alpha in alphas:
            alpha_bin = (np.abs(policy.alphas - alpha)).argmin()
            mf_returns_of_each_agent.append(result[-1]['eval_pi']['eval_mean_returns_alpha'][alpha_bin])

        mean_returns_of_each_agent = np.mean([data[i][0] for i in range(len(data))], axis=0)

    print(f'{game} {graphon} {num_players}: {np.max(np.abs(mean_returns_of_each_agent - mf_returns_of_each_agent))}', flush=True)

    if fixed_alpha:
        with open(args['experiment_directory'] + f'nagent_fixed_seeded_{id_run}_{num_players}_{beta:.2f}.pkl', 'wb') as f:
            pickle.dump(data, f, 4)
    else:
        with open(args['experiment_directory'] + f'nagent_seeded_{id_run}_{num_players}_{beta:.2f}.pkl', 'wb') as f:
            pickle.dump(data, f, 4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation of policies in finite agent game")
    parser.add_argument('--num_players_point', type=int)
    parser.add_argument('--game')
    parser.add_argument('--graphon')
    parser.add_argument('--fixed_alphas', type=int)
    parser.add_argument('--id', type=int)
    parser.add_argument('--beta', type=float)
    args = parser.parse_args()

    run_config((args.num_players_point,
                args.game,
                args.graphon,
                args.fixed_alphas,
                args.id,
                args.beta))
