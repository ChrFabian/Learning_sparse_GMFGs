import subprocess
import numpy as np

if __name__ == '__main__':
    child_processes = []

    import multiprocessing
    num_cores = multiprocessing.cpu_count()

    for graphon in ['power']:
        for game in ['Cyber-Het-Graphon']:
            for solver in ['omd']:
                p = subprocess.Popen(['python',
                                      './experiments/run.py',
                                      '--game=' + game,
                                      '--solver=' + solver,
                                      '--simulator=exact',
                                      '--evaluator=exact',
                                      '--iterations=1000',
                                      '--eta=' + '%f' % 1,
                                      '--num_alphas=25',
                                      '--graphon=' + graphon,
                                      ])
                child_processes.append(p)
                if len(child_processes) >= num_cores - 2:
                    for p in child_processes:
                        p.wait()
                    child_processes = []

    for p in child_processes:
        p.wait()
