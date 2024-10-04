#!/usr/bin/env python3

import sys
sys.path.append('../')

import numpy as np
from matrix import KernelMatrix, NonsymmetricKernelMatrix
from rpcholesky import rpcholesky

# Load data
print("Loading data...")
data = np.load('data/md17_malonaldehyde.npz')
X = data["R"]
# X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
Y = data["E"]

# Transform data
X = np.sum((X[:, :, np.newaxis, :] - X[:, np.newaxis, :, :])**2, axis = -1)**.5
X = X[:, np.triu_indices(9,1)[0], np.triu_indices(9,1)[1]] ** -1.0

# Shuffle data
print("Shuffling data...")
permutation = np.random.permutation(X.shape[0])
X = X[permutation, :]
Y = Y[permutation, :]

# Sub-select data
print("Sub-selecting data...")
n_train = 10000
n_test  = 10000 # X.shape[0] - n_train
X_train = X[0:n_train, :]
X_test  = X[n_train:(n_train+n_test), :]
Y_train = Y[0:n_train, :]
Y_test  = Y[n_train:(n_train+n_test), :]

# Run RPCholesky
print("Running RPCholesky...")
A = KernelMatrix(X_train, kernel="gaussian", bandwidth = 4.0)
k = 1000
lra = rpcholesky(A, k, accelerated = True)
rows = lra.get_rows()

# Solve least-squares problem
print("Solving least-squares...")
coeffs, resids, _, _ = np.linalg.lstsq(A[:, lra.get_indices()], Y_train, rcond=None)

# Evaluate error
print("Evaluating test error...")
A_test = A.out_of_sample(X_test, lra.get_indices())
pred = A_test @ coeffs
print(np.mean(np.abs(pred - Y_test)))
