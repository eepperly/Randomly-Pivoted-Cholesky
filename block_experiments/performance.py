#!/usr/bin/env python3

import sys
sys.path.append('../')

import numpy as np
from gallery import gallery
from rpcholesky import rpcholesky
from time import time
from scipy.io import savemat

N = int(1e4)
k = int(1e3)
b = int(1e2)
max_items = np.Inf
trials = 1

methods = { "RPCholesky" : lambda A, k: rpcholesky(A, k),
            "BlockRPC" : lambda A, k: rpcholesky(A, k, b = b),
            "Accelerated" : lambda A, k: rpcholesky(A, k, b = b, accelerated = True) }

stuff_to_save = {**{ f"{name}_time" : [] for name in methods }, **{ f"{name}_error" : [] for name in methods }}

for item, info in enumerate(gallery(N)):
    if item >= max_items:
        break
    matname, A = info

    print(f"Computing a reference low-rank approximation to matrix {matname}")
    lra = rpcholesky(A, 3*k, b = 100, accelerated = True)
    lra = lra.eigenvalue_decomposition()
    Atrace = A.trace()
    best_error = (Atrace - sum(np.sort(lra.evals())[-k:])) / Atrace
    del lra
    if best_error < 1e-14:
        print(f"Error was {best_error}. This matrix is too rank-deficient, moving on")
        continue
    print(f"Done! Error was {best_error}")
    
    for name, method in methods.items():
        start = time()
        lra = method(A, k)
        stop = time()

        total_time = stop - start
        stuff_to_save[f"{name}_time"].append(total_time)

        error_ratio = (Atrace - lra.trace()) / Atrace / best_error
        stuff_to_save[f"{name}_error"].append(error_ratio)

        del lra

        print(f"{item}\t{name}\t{total_time}\t{error_ratio}")

    savemat("data/performance.mat", stuff_to_save)
