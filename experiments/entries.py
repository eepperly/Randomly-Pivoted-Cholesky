#!/usr/bin/env python3

'''
Code to test the number of entry evaluations required for 
different Nystrom methods, using the 'Smile' and 'Outliers'
test matrices This code was used to produce Figure 2 in the
manuscript together with 'matlab_plotting/make_entry_plots.m'
'''

import sys
sys.path.append('../')

import numpy as np
import dpp_lra, rpcholesky, unif_sample, leverage_score
import gallery
import scipy

np.random.seed(342384230)

n = 10000
num_trials = 100
ks = range(0, 160, 10)

matrices = { "smile" : gallery.smile(n, bandwidth = 2.0),
             "spiral" : gallery.robspiral(n) }

methods = { 'DPP' : dpp_lra.dpp_vfx,
            'RLS' : leverage_score.recursive_rls_acc,
            'Uniform' : unif_sample.uniform_sample,
            'RPCholesky' : rpcholesky.simple_rpcholesky,
            'Greedy' : rpcholesky.greedy }

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
                    A.reset()
                    print("     ", queries[idx,i] / (n*k))
            except ValueError:
                print("    error!")
                A.reset()
                queries[idx,:] = np.NaN

            output = {"queries" : queries}
            scipy.io.savemat("data/{}_{}_queries.mat".format(matrix_name,name),output)
