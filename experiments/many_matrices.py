#!/usr/bin/env python3

'''
Code to compare the approximation error of different Nystrom
methods on a bed of real datasets, printing the results as rows
of a LaTeX table. The datasets must first be downloaded by running
'download_data.py' in the root directory of the repository.
'''

import sys
sys.path.append('../')

from scipy.sparse import issparse
import os
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import dpp_lra, rpcholesky, unif_sample, leverage_score
from utils import approximation_error
from matrix import KernelMatrix

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "preprocessed")
scaler = StandardScaler()
trials = 10

methods = { 'RLS' : leverage_score.recursive_rls_acc,
            'Uniform' : unif_sample.uniform_sample,
            'RPChol' : rpcholesky.simple_rpcholesky,
            'Greedy' : rpcholesky.greedy,
            'BlockRPChol' : rpcholesky.block_rpcholesky }

print(" &", " & ".join(methods.keys()), "& $\\eta$", end = "")
for filename in os.listdir(data_folder):
    print(" \\\\")
    print(filename[:-4].ljust(15), end="")
    data = loadmat(os.path.join(data_folder, filename))
    X = data["Xtr"]
    if issparse(X):
        X = X.toarray()
    X = scaler.fit_transform(X)
    
    A = KernelMatrix(X[:min(X.shape[0],10000),:], bandwidth=np.sqrt(X.shape[1]))
    
    for method_name, method in methods.items():
        errors = np.zeros(trials)
        for i in range(trials):
            lra = method(A, 1000)
            errors[i] = approximation_error(A, lra) / A.trace()
        print(" &", "{:.2e}".format(np.median(errors)), end="")

    evals, evecs = np.linalg.eigh(A[:,:])
    evals = evals[::-1]
    print(" &", sum(evals[1000:]) / sum(evals))
