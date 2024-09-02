#!/usr/bin/env python3

import sys
sys.path.append('../')

from rpcholesky import rpcholesky
from gallery import smile, expspiral, outliers
from matrix import KernelMatrix
from time import time
import numpy as np
import scipy as sp

n = int(1e5)
k = int(1e3)
A = smile(n, bandwidth = 0.2, extra_stability=True)
# A = outliers(n)
# A = expspiral(n)
# A = KernelMatrix(np.random.rand(n,1000))

trials = 10
ks = [10] + list(range(200,1100,200))

methods = { "Accel" : lambda k: rpcholesky(A, k, accelerated = True, b = 120),
            "Block" : lambda k: rpcholesky(A, k, accelerated = False, b = 120),
            "Basic" : lambda k: simple_rpcholesky(A, k) }
times = { method : np.zeros((len(ks),trials)) for method in methods } 
errs =  { method : np.zeros((len(ks),trials)) for method in methods } 

for idx in range(len(ks)):
    k = ks[idx]
    for method_name, method in methods.items():
        for trial in range(trials):
            start = time()
            lra = method(k)
            times[method_name][idx,trial] = time() - start
            errs[method_name][idx,trial] = (A.trace() - lra.trace()) / A.trace()
            
        print("{}\t{}\t{}\t{}".format(k, method_name, np.mean(times[method_name][idx,:]), np.mean(errs[method_name][idx,:])))
    print()

    sp.io.savemat("data/initial_compare.mat", {**{"{}_times".format(method) : times[method] for method in methods}, **{"{}_errs".format(method) : errs[method] for method in methods}, "X" : A.data} )
