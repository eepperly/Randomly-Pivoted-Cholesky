#!/usr/bin/env python3

'''
Code to time different Nystrom methods when applied to
a large kernel matrix. The size of the matrix, the 
data, and the approximation rank can be set by changing
n, d, and ks.
'''

import sys
sys.path.append('../')

import numpy as np
import time as time
import dpp_lra
import rpcholesky
import unif_sample
import leverage_score
from utils import approximation_error
import time
from matrix import PSDMatrix
import gallery
import matplotlib.pyplot as plt
import scipy

n = 1000
d = 10
ks = range(0, 120, 20)
A = gallery.kernel_from_data(np.random.randn(n,d), kernel="laplace")

num_trials = 1

methods = { 'RLS' : leverage_score.recursive_rls_acc,
            'Uniform' : unif_sample.uniform_sample,
            'RPCholesky' : rpcholesky.rpcholesky,
            'Greedy' : rpcholesky.greedy,
            'BlockRPCholesky' : rpcholesky.block_rpcholesky,
            'DPP' : dpp_lra.dpp_vfx }

for name, method in methods.items():
    print(name)
    times = np.zeros((len(ks), num_trials))

    for k, idx in zip(ks, range(len(ks))):
        print(" ", k)
        if k == 0:
            continue
        
        for i in range(num_trials):
            start = time.time()
            F = method(A, k)
            times[idx, i] = time.time() - start
            print("   ", times[idx,i])

    output = {"times" : times}
    scipy.io.savemat("data/{}_l1_time.mat".format(name),output)
