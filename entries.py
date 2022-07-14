#!/usr/bin/env python3

'''
Code to test the number of entry evaluations required for 
different Nystrom methods, using the 'Smile' and 'Outliers'
test matrices This code was used to produce Figure 2 in the
manuscript together with 'matlab_plotting/make_entry_plots.m'
'''

import numpy as np
import dpp_lra, rp_cholesky, unif_sample, leverage_score
import gallery
import scipy

n = 10000
num_trials = 10
ks = range(0, 120, 20)

matrices = { "smile" : gallery.smile(n, bandwidth = 2.0),
             "outliers" : gallery.outliers(n) }

methods = { 'DPP' : dpp_lra.dpp_vfx,
            'RLS' : leverage_score.recursive_rls_acc,
            'Uniform' : unif_sample.uniform_sample,
            'RPCholesky' : rp_cholesky.rp_cholesky,
            'Greedy' : rp_cholesky.greedy }

for matrix_name, A in matrices.items():
    print(matrix_name)
    for name, method in methods.items():
        print(" ", name)
        queries = np.zeros((len(ks), num_trials))

        for k, idx in zip(ks, range(len(ks))):
            print("   ", k)
            if k == 0:
                continue

            try:
                for i in range(num_trials):
                    F = method(A, k)
                    queries[idx, i] = A.num_queries()
                    A.reset_queries()
                    print("     ", queries[idx,i] / (n*k))
            except ValueError:
                print("    error!")
                A.reset_queries()
                queries[idx,:] = np.NaN

            output = {"queries" : queries}
            scipy.io.savemat("data/{}_{}_queries.mat".format(matrix_name,name),output)
