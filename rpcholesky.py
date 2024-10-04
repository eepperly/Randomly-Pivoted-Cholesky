#!/usr/bin/env python3

import numpy as np
import scipy as sp
from lra import PSDLowRank
from warnings import warn
from time import time
from matrix import AbstractPSDMatrix, PSDMatrix

greedy_lapack = sp.linalg.get_lapack_funcs('pstrf', dtype=np.float64)

def cholesky_helper(A, k, alg, stoptol = 0):
    if not isinstance(A, AbstractPSDMatrix):
        A = PSDMatrix(A)
    
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

def _greedy_cholesky(A, rtol = None, atol = None):
    trace = np.trace(A)
    L, piv, rank, info = greedy_lapack(A, lower=True)
    L = np.tril(L)
    if not (rtol is None) and not (atol is None):
        if atol is None:
            atol = 0
        if rtol is None:
            rtol = 0
        trailing_sums = trace - np.cumsum(np.linalg.norm(L, axis=0)**2)
        rank = np.argmax(trailing_sums < (atol + rtol * trace)) + 1
    return L, piv, rank

def block_cholesky_helper(A, k, b, alg, stoptol = 1e-14, strategy = "regularize", rbrp_atol = 0.0, rbrp_rtol = "1/b"):
    if not isinstance(A, AbstractPSDMatrix):
        A = PSDMatrix(A)
    
    diags = A.diag()
    n = A.shape[0]
    orig_trace = sum(diags)
    scale = 2*max(diags)
    if stoptol is None:
        stoptol = 1e-14
    if "1/b" == rbrp_rtol:
        rbrp_rtol = 1.0/b
    
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

        elif "rbrp" == strategy or "pivoting" == strategy or "pivoted" == strategy:
            H = A[idx,idx] - G[0:counter,idx].T @ G[0:counter,idx]
            L, piv, rank = _greedy_cholesky(H, rtol = rbrp_rtol, atol = rbrp_atol)
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

def accelerated_rpcholesky(A, k, b = "auto", stoptol = 1e-13, verbose=False):
    if not isinstance(A, AbstractPSDMatrix):
        A = PSDMatrix(A)
    
    diags = A.diag()
    n = A.shape[0]
    orig_trace = sum(diags)
    if stoptol is None:
        stoptol = 1e-13

    if "auto" == b:
        b = int(np.ceil(k / 10))
        auto_b = True
    else:
        auto_b = False
    
    # row ordering
    G = np.zeros((k,n))
    rows = np.zeros((k,n))
    
    rng = np.random.default_rng()
    arr_idx = np.zeros(k, dtype=int)
    
    counter = 0
    while counter < k:
        idx = rng.choice(range(n), size = b, p = diags / sum(diags), replace=True)

        if auto_b:
            start = time()
        
        H = A[idx, idx] - G[0:counter,idx].T @ G[0:counter,idx]
        L, accepted = rejection_cholesky(H)
        num_sel = len(accepted)

        if num_sel > k - counter:
            num_sel = k - counter
            accepted = accepted[:num_sel]
            L = L[:num_sel,:num_sel]
        
        idx = idx[accepted]

        if auto_b:
            rejection_time = time() - start
            start = time()

        arr_idx[counter:counter+num_sel] = idx
        rows[counter:counter+num_sel,:] = A[idx,:]
        G[counter:counter+num_sel,:] = rows[counter:counter+num_sel,:] - G[0:counter,idx].T @ G[0:counter,:]
        G[counter:counter+num_sel,:] = np.linalg.solve(L, G[counter:counter+num_sel,:])
        diags -= np.sum(G[counter:counter+num_sel,:]**2, axis=0)
        diags = diags.clip(min = 0)

        if auto_b:
            process_time = time() - start

            # Assuming rejection_time ~ A b^2 and process_time ~ C b
            # then obtaining rejection_time = process_time / 4 entails
            # b = C / 4A = (process_time / b) / 4 (rejection_time / b^2)
            #   = b * process_time / (4 * rejection_time)
            target = int(np.ceil(b * process_time / (4 * rejection_time)))
            b = max([min([target, int(np.ceil(1.5*b)), int(np.ceil(k/3))]),
                     int(np.ceil(b/3)), 10])

        counter += num_sel

        if stoptol > 0 and sum(diags) <= stoptol * orig_trace:
            G = G[:counter,:]
            rows = rows[:counter,:]
            break

        if verbose:
            print("Accepted {} / {}".format(num_sel, b))

    return PSDLowRank(G, idx = arr_idx, rows = rows)

def rpcholesky(A, k, b = None, accelerated = True, **kwargs):
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

def simple_rpcholesky(A, k, **kwargs):
    return rpcholesky(A, k, b = None, accelerated = False, **kwargs)
    
def block_rpcholesky(A, k, b = 100, **kwargs):
    return block_cholesky_helper(A, k, b, 'rp', **kwargs)

def block_greedy(A, k, b = 100, stoptol = None):
    return block_cholesky_helper(A, k, b, 'greedy', **kwargs)
