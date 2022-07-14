#!/usr/bin/env python3

'''
Code to test the approximation error of different Nystrom
methods, using the 'Smile' and 'Outliers' test matrices.
 This code was used to produce Figure 1 in the manuscript
 together with 'matlab_plotting/make_comparison_plots.m'
'''

import numpy as np
import dpp_lra, rp_cholesky, unif_sample, leverage_score
from utils import approximation_error
from matrix import PSDMatrix
import gallery
import scipy

n = 10000
As = { "smile" : gallery.smile(n, bandwidth = 2.0),
       "outliers" : gallery.outliers(n) }
ks = range(20, 220, 20)

methods = { 'DPP' : dpp_lra.dpp_cubic,
            'RLS' : leverage_score.recursive_rls_acc,
            'Uniform' : unif_sample.uniform_sample,
            'RPCholesky' : rp_cholesky.rp_cholesky,
            'Greedy' : rp_cholesky.greedy }

num_trials = 100

for matrix_name, A in As.items():
    print(matrix_name)
    full_matrix = A[:,:]
    full_matrix_norm = scipy.linalg.eigh(full_matrix, eigvals_only = True, subset_by_index = [n-1,n-1])[0]
    for name, method in methods.items():
        print(name)
        
        trace_norm_errors = np.zeros((len(ks), num_trials))
        trace_norm_errors[:] = np.NaN
        spectral_norm_errors = np.zeros((len(ks), num_trials))
        spectral_norm_errors[:] = np.NaN
        
        for k, idx in zip(ks, range(len(ks))):
            try:
                print(k)
                for i in range(num_trials):
                    F = method(PSDMatrix(full_matrix), k)
                    spectral_norm_errors[idx,i] = scipy.linalg.eigh(full_matrix - F.matrix(), eigvals_only = True, subset_by_index = [n-1,n-1])[0] / full_matrix_norm
                    trace_norm_errors[idx,i] = approximation_error(A, F) / A.trace()

            except (np.linalg.LinAlgError, ValueError) as e:
                break

        output = {"trace_norm_errors" : trace_norm_errors, "spectral_norm_errors" : spectral_norm_errors}
        scipy.io.savemat("data/{}_{}.mat".format(matrix_name, name), output)
                
