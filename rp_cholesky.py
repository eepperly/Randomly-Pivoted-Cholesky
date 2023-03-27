#!/usr/bin/env python3

import numpy as np
from lra import PSDLowRank

def cholesky_helper(A, k, alg):
    n = A.shape[0]
    diags = A.diag()
    
    # row ordering, is much faster for large scale problems
    F = np.zeros((k,n))
    rows = np.zeros((k,n))
    rng = np.random.default_rng()
    
    arr_idx = []
    
    for i in range(k):
        if alg == 'rp':
            idx = rng.choice(range(n), p = diags / sum(diags))
        elif alg == 'greedy':
            idx = rng.choice(np.where(diags == np.max(diags))[0])
        else:
            raise RuntimeError("Algorithm '{}' not recognized".format(alg))

        arr_idx.append(idx)
        rows[i,:] = A[idx,:]
        F[i,:] = (rows[i,:] - F[:i,idx].T @ F[:i,:]) / np.sqrt(diags[idx])
        diags -= F[i,:]**2
        diags = diags.clip(min = 0)

    return PSDLowRank(F.T, idx = arr_idx, rows = rows)

def block_cholesky_helper(A, k, b, alg):
    diags = A.diag()
    n = A.shape[0]
    
    # row ordering
    F = np.zeros((k,n))
    rows = np.zeros((k,n))
    
    rng = np.random.default_rng()

    arr_idx = []
    
    cols = 0
    while cols < k:
        block_size = min(k-cols, b)
        
        if alg == 'rp':
            idx = rng.choice(range(n), size = 2*block_size, p = diags / sum(diags),replace=False)
            idx = np.unique(idx)[:block_size]
            block_size = len(idx)
        elif alg == 'greedy':
            idx = np.argpartition(diags, -block_size)[-block_size:]
        else:
            raise RuntimeError("Algorithm '{}' not recognized".format(alg))

        arr_idx.extend(idx)
        rows[cols:cols+block_size,:] = A[idx,:]
        F[cols:cols+block_size,:] = rows[cols:cols+block_size,:] - F[0:cols,idx].T @ F[0:cols,:]
        C = F[cols:cols+block_size,idx]
        L = np.linalg.cholesky(C+100*np.finfo(float).eps*np.identity(block_size))
        F[cols:cols+block_size,:] = np.linalg.solve(L, F[cols:cols+block_size,:])
        diags -= np.sum(F[cols:cols+block_size,:]**2, axis=0)
        diags = diags.clip(min = 0)

        cols += block_size

    return PSDLowRank(F.T, idx = arr_idx, rows = rows)

def rp_cholesky(A, k):
    return cholesky_helper(A, k, 'rp')

def greedy(A, k):
    return cholesky_helper(A, k, 'greedy')

def block_rp_cholesky(A, k, b = 100):
    return block_cholesky_helper(A, k, b, 'rp')

def block_greedy(A, k, b = 100):
    return block_cholesky_helper(A, k, b, 'greedy')
