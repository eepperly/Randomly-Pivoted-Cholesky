#!/usr/bin/env python3

from matrix import KernelMatrix
import numpy as np
from time import time

N = int(1e5)
bs = np.ceil(np.logspace(0,3,30))
bs = np.unique(np.array(bs, dtype=int))
trials = 100

for kernelname in ["gaussian", "laplace"]:
    for d in [1,10,100,1000]:
        X = np.random.randn(N, d)
        A = KernelMatrix(X, kernelname)
        for b in bs:
            idx = np.array(range(b))
            avg = 0
            for trial in range(trials):
                start = time()
                Y = A[:,idx]
                stop = time()
                avg += (stop - start) / trials
            print("{}\t{}\t{}\t{}".format(kernelname,d,b,avg/b))

