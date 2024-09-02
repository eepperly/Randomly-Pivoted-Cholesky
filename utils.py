#!/usr/bin/env python3

import numpy as np
from lra import NystromExtension

def lra_from_sample(A, sample):
    rows = A[sample,:]
    core = rows[:,sample]
    return NystromExtension(core, rows = rows, idx = sample)

def approximation_error(A, lra, relative=False):
    error = A.trace() - lra.trace()
    if relative:
        error /= A.trace()
    return error

class MatrixWrapper(object):

    def __init__(self, A):
        self.A = A

    def __call__(self, X, Y = None):
        if Y is None:
            Y = X
        return self.A[np.ix_(X.ravel(), Y.ravel())]

    def diag(self, X):
        return np.array([self.A[i,i] for i in X.ravel()])
