#!/usr/bin/env python3

import os, sys
from dppy.finite_dpps import FiniteDPP
import numpy as np
from utils import lra_from_sample, MatrixWrapper

def dpp_sample_helper(A, k, **params):

    n = A.shape[0]
    mode = 'alpha' if ('mode' not in params) else params['mode']

    if mode == 'alpha' or mode == 'vfx':
        X = np.array([range(n)]).T
        dpp = FiniteDPP('likelihood', False, L_eval_X_data = (MatrixWrapper(A), X))
    else:
        dpp = FiniteDPP('likelihood', False, L = A)

    sys.stderr = open(os.devnull, 'w')
    if mode == 'mcmc':
        sample = dpp.sample_mcmc_k_dpp(k)
    elif mode == 'alpha':
        sample = dpp.sample_exact_k_dpp(k, mode=mode, early_stop=True)
    else:
        sample = dpp.sample_exact_k_dpp(k, mode=mode)
    sys.stderr = sys.__stderr__

    return lra_from_sample(A, sample)
        
def dpp_cubic(A, k):
    return dpp_sample_helper(A[:,:], k, mode = 'GS')

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
