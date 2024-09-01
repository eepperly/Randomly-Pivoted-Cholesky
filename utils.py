#!/usr/bin/env python3

import numpy as np
from lra import NystromExtension
from scipy.linalg import solve_triangular

def lra_from_sample(A, sample):
    F = A[:,sample]
    C = F[sample,:]
    k = C.shape[0]
    Lc = np.linalg.cholesky(C+C.shape[0]*C.max()*np.finfo(float).eps*np.identity(k))
    factor = solve_triangular(Lc, F.T,lower=True).T     
    return NystromExtension(F, C, factor=factor, idx = sample, rows = F.T)

def approximation_error(A, lra):
    return A.trace() - lra.trace()

class MatrixWrapper(object):

    def __init__(self, A):
        self.A = A

    def __call__(self, X, Y = None):
        if Y is None:
            Y = X
        return self.A[np.ix_(X.ravel(), Y.ravel())]

    def diag(self, X):
        return np.array([self.A[i,i] for i in X.ravel()])
