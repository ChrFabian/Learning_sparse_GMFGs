import itertools
import subprocess
import numpy as np


if __name__ == '__main__':
    num_players_points = range(2, 100, 1)
    games = ['Cyber-Graphon', 'Beach-Graphon']
    graphons = ['power']
    fixed_alphas = [False]
    ids = range(1)
    num_betas = 50
    betas = np.linspace(1 / num_betas / 2, 1 - 1 / num_betas / 2, num_betas)
    inputs = itertools.product(num_players_points, games, graphons, fixed_alphas, ids, betas)

    child_processes = []

    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    core_idx = 1

    for input in inputs:
        p = subprocess.Popen([
                              # 'taskset',
                              # '--cpu-list',
                              # '%d' % core_idx,
                              'python',
                              './experiments/run_once_nagent_compare_sparse.py',
                              f'--num_players_point={input[0]}',
                              f'--game={input[1]}',
                              f'--graphon={input[2]}',
                              "--fixed_alphas=%d" % input[3],
                              f'--id={input[4]}',
                              f'--beta={input[5]}',
                              ])
        core_idx += 1
        child_processes.append(p)
        if len(child_processes) > 29:
            for p in child_processes:
                p.wait()
            child_processes = []
            core_idx = 1

    for p in child_processes:
        p.wait()