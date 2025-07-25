#!/usr/bin/env python3

import sys
sys.path.append('../')

from matrix import KernelMatrix
import numpy as np
from time import time
from scipy.io import savemat

N = int(1e5)
bs = np.ceil(np.logspace(0,3,30))
bs = np.unique(np.array(bs, dtype=int))
trials = 100
ds = [1,10,100,1000]

all_data = {}

for kernelname in ["gaussian", "laplace"]:
    data = np.zeros((len(ds)*len(bs),3))
    i = 0
    for d in ds:
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
            data[i,0] = d
            data[i,1] = b
            data[i,2] = avg/b
            i += 1

    all_data[kernelname] = data

import os
os.makedirs("data", exist_ok=True)
    
savemat("data/generation.mat", all_data)

