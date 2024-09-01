#!/usr/bin/env python3

'''
Code to show the selected landmarks for different Nystrom
methods, using the 'Smile' and 'Outliers' test matrices.
This code was used to produce Figure 1 in the manuscript
together with 'matlab_plotting/make_chosen_plots.m'
'''

import numpy as np
import gallery
import scipy
from rp_cholesky import rp_cholesky, greedy
from unif_sample import uniform_sample
from leverage_score import recursive_rls_acc
from utils import approximation_error

n = 10000
k = 40

np.random.seed(342384230)

# Add matrices
As = { "smile" : gallery.smile(n, bandwidth = 2.0),
       "spiral" : gallery.robspiral(n) }

methods = { 'RLS' : recursive_rls_acc,
            'Uniform' : uniform_sample,
            'RPCholesky' : rp_cholesky,
            'Greedy' : greedy }

for matrix_name, A in As.items():
    for algorithm_name, algorithm in methods.items():
        print(matrix_name, algorithm_name)
        lra = algorithm(A, k)
        print(approximation_error(A, lra) / A.trace())
        scipy.io.savemat("data/{}_{}_picked.mat".format(matrix_name, algorithm_name), {"picked" : np.array(lra.idx)+1, "X" : A.data})
