#!/usr/bin/env python3

import os, sys
import numpy as np

# dppy still uses aliases (np.float, np.int, ...) that were removed in numpy 1.24
for _name, _type in [("float", float), ("int", int), ("bool", bool), ("complex", complex), ("object", object)]:
    if _name not in vars(np):
        setattr(np, _name, _type)

from dppy.finite_dpps import FiniteDPP
from utils import lra_from_sample, MatrixWrapper

def dpp_sample_helper(A, k, **params):

    n = A.shape[0]
    mode = 'alpha' if ('mode' not in params) else params['mode']

    if A.dpp_stuff is None or A.dpp_stuff[1] != k:        
        if mode == 'alpha' or mode == 'vfx':
            X = np.array([range(n)]).T
            A.dpp_stuff = (FiniteDPP('likelihood', False, L_eval_X_data = (MatrixWrapper(A), X)), k)
        else:
            A.dpp_stuff = (FiniteDPP('likelihood', False, L = A[:,:]), k)

    # Suppress dppy's progress output on stderr, restoring it even if sampling fails
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        if mode == 'mcmc':
            sample = A.dpp_stuff[0].sample_mcmc_k_dpp(k)
        elif mode == 'alpha':
            sample = A.dpp_stuff[0].sample_exact_k_dpp(k, mode=mode, early_stop=True)
        else:
            sample = A.dpp_stuff[0].sample_exact_k_dpp(k, mode=mode)
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

    return lra_from_sample(A, sample)
        
def dpp_cubic(A, k):
    return dpp_sample_helper(A, k, mode = 'GS')

def dpp_vfx(A, k):
    return dpp_sample_helper(A, k, mode = 'vfx')

def dpp_alpha(A, k):
    return dpp_sample_helper(A, k, mode = 'alpha')

def dpp_mcmc(A, k):
    return dpp_sample_helper(A, k, mode = 'mcmc')

if __name__ == "__main__":
    from gallery import smile
    A = smile(1000)
    dpp_vfx(A, 20)
