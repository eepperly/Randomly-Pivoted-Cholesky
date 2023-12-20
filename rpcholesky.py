#!/usr/bin/env python3

import numpy as np
import scipy as sp
from lra import PSDLowRank
from warnings import warn

def cholesky_helper(A, k, alg, stoptol = 0):
    n = A.shape[0]
    diags = A.diag()
    orig_trace = sum(diags)
    if stoptol is None:
        stoptol = 0
    
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

        if stoptol > 0 and sum(diags) <= stoptol * orig_trace:
            G = G[:i,:]
            rows = rows[:i,:]
            break

    return PSDLowRank(G, idx = arr_idx, rows = rows)

def block_cholesky_helper(A, k, b, alg, stoptol = 1e-14):
    diags = A.diag()
    n = A.shape[0]
    orig_trace = sum(diags)
    scale = 2*max(diags)
    if stoptol is None:
        stoptol = 1e-14
    
    # row ordering
    G = np.zeros((k,n))
    rows = np.zeros((k,n))
    
    rng = np.random.default_rng()

    arr_idx = []
    
    counter = 0
    while counter < k:
        block_size = min(k-counter, b)
        
        if alg == 'rp':
            idx = rng.choice(range(n), size = 2*block_size, p = diags / sum(diags), replace = True)
            idx = np.unique(idx)[:block_size]
            block_size = len(idx)
        elif alg == 'greedy':
            idx = np.argpartition(diags, -block_size)[-block_size:]
        else:
            raise RuntimeError("Algorithm '{}' not recognized".format(alg))

        arr_idx.extend(idx)
        rows[counter:counter+block_size,:] = A[idx,:]
        G[counter:counter+block_size,:] = rows[counter:counter+block_size,:] - G[0:counter,idx].T @ G[0:counter,:]
        C = G[counter:counter+block_size,idx]
        try:
            L = np.linalg.cholesky(C + np.finfo(float).eps*b*scale*np.identity(block_size))
            G[counter:counter+block_size,:] = np.linalg.solve(L, G[counter:counter+block_size,:])
        except np.linalg.LinAlgError:
            warn("Cholesky failed in block partial Cholesky. Falling back to eigendecomposition")
            evals, evecs = np.linalg.eigh(C)
            evals[evals > 0] = evals[evals > 0] ** (-0.5)
            evals[evals < 0] = 0
            print(counter,G.shape, evecs.shape, evals.shape)
            G[counter:counter+block_size,:] = evals[:,np.newaxis] * (evecs.T @ G[counter:counter+block_size,:])
        diags -= np.sum(G[counter:counter+block_size,:]**2, axis=0)
        diags = diags.clip(min = 0)

        counter += block_size

        if stoptol > 0 and sum(diags) <= stoptol * orig_trace:
            G = G[:counter,:]
            rows = rows[:counter,:]
            break

    return PSDLowRank(G, idx = arr_idx, rows = rows)

def rejection_cholesky(H):
    b = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise RuntimeError("rejection_cholesky requires a square matrix")
    u = np.array([H[j,j] for j in range(b)])

    idx = []
    L = np.zeros((b,b))
    for j in range(b):
        if np.random.rand() * u[j] < H[j,j]:
            idx.append(j)
            L[j:,j] = H[j:,j] / np.sqrt(H[j,j])
            H[(j+1):,(j+1):] -= np.outer(L[(j+1):,j], L[(j+1):,j])
    idx = np.array(idx)
    L = L[np.ix_(idx,idx)]
    return L, idx

def accelerated_rpcholesky(A, k, b = 100, stoptol = 1e-14):
    diags = A.diag()
    n = A.shape[0]
    orig_trace = sum(diags)
    if stoptol is None:
        stoptol = 1e-14
    
    # row ordering
    G = np.zeros((k,n))
    rows = np.zeros((k,n))
    
    rng = np.random.default_rng()
    arr_idx = np.zeros(k)
    
    counter = 0
    while counter < k:
        idx = rng.choice(range(n), size = b, p = diags / sum(diags), replace=True)

        H = A[idx, idx] - G[0:counter,idx].T @ G[0:counter,idx]
        L, accepted = rejection_cholesky(H)
        num_sel = len(accepted)
        print(num_sel)

        if num_sel > k - counter:
            num_sel = k - counter
            accepted = accepted[:num_sel]
            L = L[:num_sel,:num_sel]
        
        idx = idx[accepted]

        arr_idx[counter:counter+num_sel] = idx
        rows[counter:counter+num_sel,:] = A[idx,:]
        G[counter:counter+num_sel,:] = rows[counter:counter+num_sel,:] - G[0:counter,idx].T @ G[0:counter,:]
        G[counter:counter+num_sel,:] = np.linalg.solve(L, G[counter:counter+num_sel,:])
        diags -= np.sum(G[counter:counter+num_sel,:]**2, axis=0)
        diags = diags.clip(min = 0)

        counter += num_sel

        if stoptol > 0 and sum(diags) <= stoptol * orig_trace:
            G = G[:counter,:]
            rows = rows[:counter,:]
            break

    return PSDLowRank(G, idx = arr_idx, rows = rows)

def rpcholesky(A, k, b = None, accelerated = False, stoptol = None):
    if b is None:
        if accelerated:
            return accelerated_rpcholesky(A, k, stoptol = stoptol)
        else:
            return cholesky_helper(A, k, 'rp', stoptol = stoptol)
    elif accelerated:
        return accelerated_rpcholesky(A, k, b, stoptol = stoptol)
    else:
        return block_cholesky_helper(A, k, b, 'rp', stoptol = stoptol)

def greedy(A, k, randomized_tiebreaking = False, b = 1, stoptol = None):
    if b == 1:
        return cholesky_helper(A, k, 'rgreedy' if randomized_tiebreaking else 'greedy', stoptol = stoptol)
    else:
        if randomized_tiebreaking:
            warn("Randomized tiebreaking not implemented for block greedy method")
        return block_cholesky_helper(A, k, b, 'greedy', stoptol = stoptol)

def block_rpcholesky(A, k, b = 100, stoptol = None):
    return block_cholesky_helper(A, k, b, 'rp', stoptol = stoptol)

def block_greedy(A, k, b = 100, stoptol = None):
    return block_cholesky_helper(A, k, b, 'greedy', stoptol = stoptol)
