#!/usr/bin/env python3

import numpy as np
from recursive_nystrom import recursiveNystrom
from bless import bless, get_nystrom_embeddings
from utils import lra_from_sample, MatrixWrapper

class RecursiveNystromWrapper(object):

    def __init__(self, A):
        self.A = A
        self.shape = A.shape

    def __call__(self, X, Y = None):
        if Y is None:
            to_return = self.A.diag(X)
            return np.reshape(to_return, (to_return.shape[0], 1))
        else:
            to_return = self.A[np.ix_(X.ravel(), Y.ravel())]
            if len(to_return.shape) == 1:
                return np.reshape(to_return, (to_return.shape[0], 1))
            else:
                return to_return

def recursive_rls_helper(A, k, accelerated_flag):
    n = A.shape[0]
    sample = recursiveNystrom(np.array([range(n)]).T, k, RecursiveNystromWrapper(A), accelerated_flag)
    return lra_from_sample(A, sample)

def recursive_rls(A, k):
    return recursive_rls_helper(A, k, False)

def recursive_rls_acc(A, k):
    return recursive_rls_helper(A, k, True)

def bless_rls(A, k):
    n = A.shape[0]
    sample = bless(np.array([range(n)]).T, MatrixWrapper(A), 50.0, verbose=False, force_cpu=True).idx
    return lra_from_sample(A, sample), sample

def exact_RLS(A, k, lamb):
    n = A.shape[0]
    K = A[:,:]
    score = np.diag(np.linalg.solve(K+lamb*np.eye(n), K))
    score = score / sum(score)
    rng = np.random.default_rng()
    sample = rng.choice(range(n), size=k, p = score, replace=False)

    return lra_from_sample(A, sample)
