#!/usr/bin/env python3

import sys
sys.path.append('../')

import numpy as np
from rpcholesky import rpcholesky
from gallery import smile
from utils import approximation_error
from time import time

n = int(1e5)
k = int(1e3)
A = smile(n, bandwidth = 0.2, extra_stability=True)

methods = { "Acc"  : lambda: rpcholesky(A, k, b = 120, accelerated=True),
            "Block"  : lambda: rpcholesky(A, k, b = 120, accelerated=False),
            "RBRP" : lambda: rpcholesky(A, k, b = 120, accelerated=False, strategy="rbrp") }
trials = 100

for name, method in methods.items():
    print(name)
    timespent = 0.0
    errors = np.zeros(trials)
    times  = np.zeros(trials)
    for trial in range(trials):
        print(trial)
        start = time()
        lra = method()
        times[trial]  = time() - start
        errors[trial] = approximation_error(A, lra, relative=True)
    print()
    print("{}\t{} +/- {}\t{} +/- {}".format(name, np.mean(errors), np.std(errors),
                                            np.mean(times), np.std(times)))

