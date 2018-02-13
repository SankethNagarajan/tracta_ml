import tracta_ml.genetic_operators as gen_ops
from joblib import Parallel, delayed
import multiprocessing
import numpy as np


def trmnt_selct(parents):
    k = 2  #Tournament size
    ind_list = [i for i in range(len(parents))]
    pair = np.random.choice(ind_list, size=k, replace=False)
    if parents[pair[0]] > parents[pair[1]]:
        return pair[0]
    return pair[1]


def select_cross_pair(parents):
    parent1 = trmnt_selct(parents)
    while True:
        parent2 = trmnt_selct(parents)
        if parent1 != parent2:
            break
    return {1: parents[parent1], 2: parents[parent2]}


def gen_random_parents(X, Y, mod, param_dict, cv, scoring, pool_size=5):
    num_cores = multiprocessing.cpu_count()
    parents = Parallel(n_jobs=num_cores)(delayed(gen_ops.generate_parent)(X, Y, mod, param_dict, cv, scoring)\
                                         for _ in range(pool_size))
    return parents


def crossover_pool(parents, X, Y, mod, cv, scoring, pool_size):
    num_cores = multiprocessing.cpu_count()
    children = Parallel(n_jobs=num_cores)(delayed(gen_ops.crossover)(X, Y, mod, cv, scoring, select_cross_pair(parents))\
                                          for _ in range(pool_size))
    return children
