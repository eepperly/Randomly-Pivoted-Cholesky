#!/usr/bin/env python3

import numpy as np
import scipy as sp
from lra import PSDLowRank
from warnings import warn

greedy_lapack = sp.linalg.get_lapack_funcs('pstrf', dtype=np.float64)

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

def _greedy_cholesky(A, tol = None):
    L, piv, rank, info = greedy_lapack(A, lower=True)
    L = np.tril(L)
    if not (tol is None):
        index = np.where(np.abs(np.diagonal(L)) < np.sqrt(tol))[0]
        if len(index) > 0:
            rank = index[0]
    return L, piv, rank

def block_cholesky_helper(A, k, b, alg, stoptol = 1e-14, strategy = "regularize"):
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

        if "regularize" == strategy or "regularized" == strategy:
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
                G[counter:counter+block_size,:] = evals[:,np.newaxis] * (evecs.T @ G[counter:counter+block_size,:])

        elif "pivoting" == strategy or "pivoted" == strategy:
            H = A[idx,idx] - G[0:counter,idx].T @ G[0:counter,idx]
            # L, piv, rank = _greedy_cholesky(H, tol = np.finfo(float).eps*b*scale)
            L, piv, rank = _greedy_cholesky(H, tol = 1e-14)
            idx = idx[piv[0:rank]-1]
            arr_idx.extend(idx)
            rows[counter:counter+len(idx),:] = A[idx,:]
            G[counter:counter+len(idx),:] = rows[counter:counter+len(idx),:] - G[0:counter,idx].T @ G[0:counter,:]            
            G[counter:counter+len(idx),:] = np.linalg.solve(L[0:rank,0:rank], G[counter:counter+len(idx),:])

        else:
            raise ValueError("'{}' is not a valid strategy for block RPCholesky".format(strategy))
                
        diags -= np.sum(G[counter:counter+len(idx),:]**2, axis=0)
        diags = diags.clip(min = 0)

        counter += len(idx)

        if stoptol > 0 and sum(diags) <= stoptol * orig_trace:
            G = G[:counter,:]
            rows = rows[:counter,:]
            break

    return PSDLowRank(G, idx = arr_idx, rows = rows)

def rejection_cholesky(H):
    b = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise RuntimeError("rejection_cholesky requires a square matrix")
    if np.trace(H) <= 0:
        raise RuntimeError("rejection_cholesky requires a strictly positive trace")
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

def accelerated_rpcholesky(A, k, b = 100, stoptol = 1e-13):
    diags = A.diag()
    n = A.shape[0]
    orig_trace = sum(diags)
    if stoptol is None:
        stoptol = 1e-13
    
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

def rpcholesky(A, k, b = None, accelerated = False, **kwargs):
    if b is None:
        if accelerated:
            return accelerated_rpcholesky(A, k, **kwargs)
        else:
            return cholesky_helper(A, k, 'rp', **kwargs)
    elif accelerated:
        return accelerated_rpcholesky(A, k, b, **kwargs)
    else:
        return block_cholesky_helper(A, k, b, 'rp', **kwargs)

def greedy(A, k, randomized_tiebreaking = False, b = 1, **kwargs):
    if b == 1:
        return cholesky_helper(A, k, 'rgreedy' if randomized_tiebreaking else 'greedy', **kwargs)
    else:
        if randomized_tiebreaking:
            warn("Randomized tiebreaking not implemented for block greedy method")
        return block_cholesky_helper(A, k, b, 'greedy', **kwargs)

def block_rpcholesky(A, k, b = 100, **kwargs):
    return block_cholesky_helper(A, k, b, 'rp', **kwargs)

def block_greedy(A, k, b = 100, stoptol = None):
    return block_cholesky_helper(A, k, b, 'greedy', **kwargs)
