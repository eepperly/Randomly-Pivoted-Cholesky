#!/usr/bin/env python3

from scipy.sparse import issparse
import os
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import dpp_lra, rp_cholesky, unif_sample, leverage_score
from utils import approximation_error
from matrix import KernelMatrix

data_folder = os.path.join(os.getcwd(), "data/preprocessed")
scaler = StandardScaler()
trials = 10

methods = { 'RLS' : leverage_score.recursive_rls_acc,
            'Uniform' : unif_sample.uniform_sample,
            'RPChol' : rp_cholesky.rp_cholesky,
            'Greedy' : rp_cholesky.greedy,
            'BlockRPChol' : rp_cholesky.block_rp_cholesky }

print(" &", " & ".join(methods.keys()), "& $\eta$", end = "")
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
