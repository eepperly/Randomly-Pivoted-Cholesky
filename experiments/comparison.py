#!/usr/bin/env python3

'''
Code to test the approximation error of different Nystrom
methods, using the 'Smile' and 'Outliers' test matrices.
 This code was used to produce Figure 1 in the manuscript
 together with 'matlab_plotting/make_comparison_plots.m'
'''

import sys
sys.path.append('../')

import numpy as np
import dpp_lra, rpcholesky, unif_sample, leverage_score
from utils import approximation_error
from matrix import PSDMatrix
import gallery
import scipy

np.random.seed(2940938240)

n = 10000
As = { "smile" : gallery.smile(n, bandwidth = 2.0),
       "spiral" : gallery.robspiral(n) }
ks = range(10, 160, 10)

methods = { 'DPP' : dpp_lra.dpp_cubic,
            'RLS' : leverage_score.recursive_rls_acc,
            'Uniform' : unif_sample.uniform_sample,
            'RPCholesky' : rpcholesky.rpcholesky,
            'Greedy' : rpcholesky.greedy }

num_trials = 100

for matrix_name, A in As.items():
    print(matrix_name)
    A = PSDMatrix(A[:,:])
    for name, method in methods.items():
        print(name)
        
        trace_norm_errors = np.zeros((len(ks), num_trials))
        trace_norm_errors[:] = np.NaN
        
        for k, idx in zip(ks, range(len(ks))):
            try:
                print(k)
                for i in range(num_trials):
                    F = method(A, k)
                    trace_norm_errors[idx,i] = approximation_error(A, F) / A.trace()

            except (np.linalg.LinAlgError, ValueError) as e:
                break

        output = {"trace_norm_errors" : trace_norm_errors}
        scipy.io.savemat("data/{}_{}.mat".format(matrix_name, name), output)
                
