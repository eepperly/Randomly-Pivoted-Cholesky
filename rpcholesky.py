#!/usr/bin/env python3

import numpy as np
import scipy as sp
from lra import PSDLowRank
from warnings import warn

def cholesky_helper(A, k, alg):
    n = A.shape[0]
    diags = A.diag()
    
    # row ordering, is much faster for large scale problems
    G = np.zeros((k,n))
    rows = np.zeros((k,n))
    rng = np.random.default_rng()
    
    arr_idx = []
    
    for i in range(k):
        if alg == 'rp':
            idx = rng.choice(range(n), p = diags / sum(diags))
        elif alg == 'rgreedy':
            idx = rng.choice(np.where(diags == np.max(diags))[0])
        elif alg == "greedy":
            idx = np.argmax(diags)
        else:
            raise RuntimeError("Algorithm '{}' not recognized".format(alg))

        arr_idx.append(idx)
        rows[i,:] = A[idx,:]
        G[i,:] = (rows[i,:] - G[:i,idx].T @ G[:i,:]) / np.sqrt(diags[idx])
        diags -= G[i,:]**2
        diags = diags.clip(min = 0)

    return PSDLowRank(G, idx = arr_idx, rows = rows)

def block_cholesky_helper(A, k, b, alg):
    diags = A.diag()
    n = A.shape[0]
    
    # row ordering
    G = np.zeros((k,n))
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
        G[cols:cols+block_size,:] = rows[cols:cols+block_size,:] - G[0:cols,idx].T @ G[0:cols,:]
        C = G[cols:cols+block_size,idx]
        L = np.linalg.cholesky(C+np.finfo(float).eps*np.trace(C)*np.identity(block_size))
        G[cols:cols+block_size,:] = np.linalg.solve(L, G[cols:cols+block_size,:])
        diags -= np.sum(G[cols:cols+block_size,:]**2, axis=0)
        diags = diags.clip(min = 0)

        cols += block_size

    return PSDLowRank(G, idx = arr_idx, rows = rows)

def rpcholesky(A, k, b = 1):
    if b == 1:
        return cholesky_helper(A, k, 'rp')
    else:
        return block_cholesky_helper(A, k, b, 'rp')

def greedy(A, k, randomized_tiebreaking = False, b = 1):
    if b == 1:
        return cholesky_helper(A, k, 'rgreedy' if randomized_tiebreaking else 'greedy')
    else:
        if randomized_tiebreaking:
            warn("Randomized tiebreaking not implemented for block greedy method")
        return block_cholesky_helper(A, k, b, 'greedy')

def block_rpcholesky(A, k, b = 100):
    return block_cholesky_helper(A, k, b, 'rp')

def block_greedy(A, k, b = 100):
    return block_cholesky_helper(A, k, b, 'greedy')
