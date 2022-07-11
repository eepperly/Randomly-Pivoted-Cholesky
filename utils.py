#!/usr/bin/env python3

import numpy as np
from lra import NystromExtension 

def lra_from_sample(A, sample):
    F = A[:,sample]
    C = F[sample,:]
    return NystromExtension(F, C)

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
