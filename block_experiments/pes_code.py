# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code was written by Robert J. Webber, and uses
# a different implementation of RPCholesky methods then
# the rest of this paper

# =====
# SETUP
# =====

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.linalg import solve_triangular
from scipy.linalg import eigvalsh
from scipy.linalg import qr
from numba import njit
import tables

# pretty plots
colors = np.array(['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'])
font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : 16}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 1.5

#%%

# ======
# KERNEL
# ======

@njit
def kernel(X: np.ndarray, Y: np.ndarray=None):
    if Y is None:
        result = np.ones((X.shape[0], 1))
    else:
        X_squares = np.sum(np.square(X), 1)[:, np.newaxis]
        Y_squares = np.sum(np.square(Y), 1)[np.newaxis, :]
        result = X_squares + Y_squares - 2 * X @ Y.T
        # result = np.clip(result, 0, None)
        # result = np.exp(-result/2)
        result = (5 * np.clip(result, 0, None)) ** .5
        result = (1.0 + result + result**2 / 3.0) * np.exp(-result)
    return result

# from scipy.spatial import distance_matrix
# def kernel(X, Y):
#     result = distance_matrix(X, Y, p=1)
#     result = np.exp(-result)
#     return result

#%%

# =======================
# LOW-RANK APPROXIMATIONS
# =======================

# Basic RPC
def rpc(X, k=1000):
    n = X.shape[0]
    G = np.zeros((k,n))
    S = np.array([], dtype=int)
    d = np.ones(n)
    for i in range(k):
        if i % 1000 == 0:
            print('Sample', i)
        idx = np.random.choice(n, p=d/np.sum(d))
        S = np.append(S, idx)
        row = kernel(X[idx, :][np.newaxis, :], X)
        G[i,:] = (row - G[:i,idx].T @ G[:i,:]) / np.sqrt(d[idx])
        d -= G[i,:]**2
        d = d.clip(min = 0)
    return S, G

# Greedy selection
def greedy(X, k=1000):
    n = X.shape[0]
    G = np.zeros((k,n))
    S = np.array([], dtype=int)
    d = np.ones(n)
    for i in range(k):
        if i % 1000 == 0:
            print('Sample', i)
        idx = np.random.choice(np.where(d == np.max(d))[0])
        S = np.append(S, idx)
        row = kernel(X[idx, :][np.newaxis, :], X)
        G[i,:] = (row - G[:i,idx].T @ G[:i,:]) / np.sqrt(d[idx])
        d -= G[i,:]**2
        d = d.clip(min = 0)
    return S, G

# Random Fourier features (assumes Matern 5/2 kernel)
def rff(X, k=1000, nu=5/2):
    # rejection sampling
    samples = np.random.normal(size = (k, X.shape[1]))
    norms = np.random.beta(nu, X.shape[1]/2, size=k)
    norms = np.sqrt(2*nu / norms - 2*nu)
    norms /= np.sqrt(np.sum(np.square(samples), axis=1))
    samples *= norms[:, np.newaxis]
    G = samples @ X.T
    G = G + np.random.uniform(high=2*np.pi, size=k)[:, np.newaxis]
    G = np.sqrt(2 / k) * np.cos(G)
    return G

# Accelerated RPCholesky
def rpcholesky(X, k=1000, b=100):
    n = X.shape[0]
    G = np.zeros((k, n))
    S = np.array([], dtype=int)
    d = np.ones(n)
    while S.size < k:
        now = time.time()
        vals = np.random.choice(n, b, p=d/np.sum(d))
        # build residual b x b matrix
        core = kernel(X[vals, :], X[vals, :])
        core -= G[:S.size, vals].T @ G[:S.size, vals]
        # rejection sampling RPCholesky
        idx = np.array([], dtype=int)
        L_block = np.zeros((b, b))
        for i in range(b):
            if np.random.uniform() * d[vals[i]] < core[i, i]:
                new = core[i:, i] / np.sqrt(core[i, i])
                L_block[i, i:] = new
                core[i:, i:] = core[i:, i:] - np.outer(new, new)
                idx = np.append(idx, i)
            if S.size + idx.size >= k:
                break
        S2 = np.append(S, idx)
        if idx.size > 0:
            L_new = L_block[np.ix_(idx, idx)].T
            # build residual k x N wide matrix
            wide = kernel(X[vals[idx], :], X)
            wide -= G[:S.size, vals[idx]].T @ G[:S.size, :]
            # solve linear system
            wide = solve_triangular(L_new, wide, overwrite_b=True, 
                                    check_finite=False, lower=True)
            G[S.size:S2.size, :] = wide
        if idx.size > 0 and S2.size < k:
            # perform updates
            d -= np.sum(wide**2, axis = 0)
            np.clip(d, 0, None, out=d)
        print('Time:', time.time() - now)
        print(idx.size, 'out of', b, 'proposals')
        S = np.append(S, vals[idx])
    return S, G

#  Low-memory RPCholesky
def rpcholesky_lowmem(X, k=1000, b=100, mem=0):
    
    # allocate memory
    n = X.shape[0]
    if mem == 0:
        mem = k**2
    blocks = np.append(np.arange(n, step=(mem // b)), n)
    L = np.zeros((k, k))
    S = np.array([], dtype=int)
    d = np.ones(n) # just one block of size n
    while S.size < k:
        now = time.time()

        # propose block of columns
        vals = np.random.choice(n, b, p=d/np.sum(d))
        S2 = np.append(S, vals)

        # build b x b core matrix O(k^2 b) operations
        builder = kernel(X[S2, :], X[vals, :])
        if S.size == 0:
            core = builder
        else:
            tall = solve_triangular(L[:S.size, :S.size], builder[:S.size],
                                    overwrite_b=True, check_finite=False,
                                    lower=True)
            core = builder[S.size:, :] - tall.T @ tall
        
        # rejection sampling RPCholesky O(b^3) operations
        idx = np.array([], dtype=int)
        L_block = np.zeros((b, b))
        for i in range(b):
            if np.random.uniform() * d[vals[i]] < core[i, i]:
                col = core[i, i:] / np.sqrt(core[i, i])
                L_block[i, i:] = col
                core[i:, i:] = core[i:, i:] - np.outer(col, col)
                idx = np.append(idx, i)
            if S.size + idx.size >= k:
                break
        S2 = np.append(S, vals[idx])
        print('First half:', time.time() - now)
        now = time.time()

        # update Cholesky factor
        if idx.size > 0 and S2.size < k:
            L_new = L_block[np.ix_(idx, idx)].T
            if S.size == 0:
                L[:S2.size, :S2.size] = L_new
            else:
                L[S.size:S2.size, :S2.size] = \
                    np.column_stack((tall[:, idx].T, L_new))
    
            # update diagonal O(k b n) operations
            if S.size > 0:
                shorten = solve_triangular(L[:S.size, :S.size].T, tall[:, idx],
                                           overwrite_b=True, check_finite=False,
                                           lower=False).T
            for i in range(blocks.size - 1):
                blocker = kernel(X[S2, :], X[blocks[i]:blocks[i + 1], :])
                if S.size == 0:
                    wide = blocker
                else:
                    wide = blocker[S.size:, :] - shorten @ blocker[:S.size, :]
                wide = solve_triangular(L_new, wide, overwrite_b=True,
                                        check_finite=False, lower=True)
                d[blocks[i]:blocks[i+1]] = d[blocks[i]:blocks[i+1]] \
                    - np.sum(wide**2, axis = 0)
            np.clip(d, 0, None, out=d)
        
        # report progress
        S = S2
        print('Second half:', time.time() - now)
        print(idx.size, 'out of', b, 'proposals')

    return S

# Ridge leverage scores
# from https://github.com/axelv/recursive-nystrom/blob/master/recursive_nystrom.py

def recursiveNystrom(X, n_components: int, kernel_func=kernel, random_state=None, lmbda_0=0, return_leverage_score=False):

    # set up parameters
    rng = np.random.RandomState(random_state)
    n_oversample = np.log(n_components)
    k = np.ceil(n_components / (4 * n_oversample)).astype(int)
    n_levels = np.ceil(np.log(X.shape[0] / n_components) / np.log(2)).astype(int)
    perm = rng.permutation(X.shape[0])

    # set up sizes for recursive levels
    size_list = [X.shape[0]]
    for l in range(1, n_levels+1):
        size_list += [np.ceil(size_list[l - 1] / 2).astype(int)]

    # indices of points selected at previous level of recursion
    # at the base level it's just a uniform sample of ~ n_component points
    sample = np.arange(size_list[-1])
    indices = perm[sample]
    weights = np.ones((indices.shape[0],))

    # we need the diagonal of the whole kernel matrix, so compute upfront
    k_diag = kernel_func(X)

    # Main recursion, unrolled for efficiency
    for l in reversed(range(n_levels)):
        # indices of current uniform sample
        current_indices = perm[:size_list[l]]
        # build sampled kernel

        # all rows and sampled columns
        KS = kernel_func(X[current_indices,:], X[indices,:])
        SKS = KS[sample, :] # sampled rows and sampled columns

        # optimal lambda for taking O(k log(k)) samples
        if k >= SKS.shape[0]:
            # for the rare chance we take less than k samples in a round
            lmbda = 10e-6
            # don't set to exactly 0 to avoid stability issues
        else:
            # eigenvalues equal roughly the number of points per cluster, maybe this should scale with n?
            # can be interpret as the zoom level
            lmbda = (np.sum(np.diag(SKS) * (weights ** 2))
                    - np.sum(eigvalsh(SKS * weights[:,None] * weights[None,:], eigvals=(SKS.shape[0]-k, SKS.shape[0]-1))))/k
        lmbda = np.maximum(lmbda_0 * SKS.shape[0], lmbda)
        if lmbda == lmbda_0 * SKS.shape[0]:
            print("Set lambda to %d." % lmbda)

        # compute and sample by lambda ridge leverage scores
        R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
        R = np.matmul(KS, R)
        if l != 0:
            leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is n_components in expectation
            sample = np.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample
            if sample.size == 0:
                leverage_score[:] = n_components / size_list[l]
                sample = rng.choice(size_list[l], size=n_components, replace=False)
            weights = np.sqrt(1. / leverage_score[sample])

        else:
            leverage_score = np.minimum(1.0, (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            p = leverage_score/leverage_score.sum()

            sample = rng.choice(X.shape[0], size=n_components, replace=False, p=p)
        indices = perm[sample]

    if return_leverage_score:
        return indices, leverage_score[np.argsort(perm)]
    else:
        return indices

#%%

# ============
# PREPARE DATA
# ============

# Load data
# data = np.load('md17_aspirin.npz')
# data = np.load('md17_benzene2017.npz')
# data = np.load('md17_ethanol.npz')
# data = np.load('md17_malonaldehyde.npz')
# data = np.load('md17_naphthalene.npz')
# data = np.load('md17_salicylic.npz')
# data = np.load('md17_toluene.npz')
data = np.load('md17_uracil.npz')
R = data['R']
E = data['E']
del data

# Transform data
n_atoms = R.shape[1]
X = np.sum((R[:, :, np.newaxis, :] - R[:, np.newaxis, :, :])**2, axis = -1)**.5
X = X[:, np.triu_indices(n_atoms, 1)[0], np.triu_indices(n_atoms, 1)[1]] ** -1.

# Shuffle data
np.random.seed(42)
permutation = np.random.permutation(X.shape[0])
X = X[permutation, :]
Y = E[permutation, 0]

# Sub-select data
n_train = 100000
n_test  = 10000
X_train = X[:n_train, :]
X_test  = X[n_train:(n_train+n_test), :]
Y_train = Y[:n_train]
Y_test  = Y[n_train:(n_train+n_test)]

# normalization
mean = np.mean(X_train, axis=0)
X_train = X_train - mean[np.newaxis, :]
var = np.mean(X_train**2, axis=0)
X_train = X_train / var[np.newaxis, :]**.5
X_test = (X_test - mean[np.newaxis, :]) / var[np.newaxis, :]**.5
ave = Y_train.mean()
Y_train = Y_train - ave

# range
print('SD:', np.mean(Y_train**2)**.5)
print('Range:', np.max(Y_train) - np.min(Y_train))

#%%

# ==================
# PRECONDITIONED KRR
# ==================

# set up problem parameters
k = 10000 # rank of approximation
block = k // 2 # block size in accelerated RPC
mu = 1e-4 # regularization parameter
tol = 1e-3 # tolerance for relative residual in PCG
n_train, d = X_train.shape
sigma = np.sqrt(d)
Z_train = X_train / sigma
Z_test = X_test / sigma
tall = kernel(Z_test, Z_train)

# memory parameters
mem = int(1e9)
blocks = np.append(np.arange(n_train, step=mem // n_train), n_train)

# make kernel matrix
now = time.time()
h5f_A = tables.open_file('kernel_mat.h5', 'w')
filters = tables.Filters(complevel=5, complib='blosc')
A = h5f_A.create_carray(h5f_A.root, 'CArray', tables.Float64Atom(), 
                        shape=(n_train,n_train), filters=filters)
for i in range(blocks.size - 1):
    print('Rows ', blocks[i], ' to ', blocks[i+1])
    A[blocks[i]:blocks[i + 1], :] = kernel(Z_train[blocks[i]:blocks[i + 1], :], Z_train)
kernel_time = time.time() - now
print('Kernel time:', kernel_time)

# # close and open commands
# h5f_A.close()
# h5f_A = tables.open_file('kernel_mat.h5', 'r')
# A = h5f_A.root['CArray']

# set up results
methods = 5
max_iters = 1000
prep = np.zeros((methods, 2))
resid = np.zeros((methods, max_iters + 1))
mae = np.zeros((methods, max_iters + 1))
rmse = np.zeros((methods, max_iters + 1))
times = np.zeros((methods, max_iters + 1))
for j in range(4, methods):
    np.random.seed(43)
    if j == 0:
        # RPC Nystrom
        print('Running accelerated RPCholesky...')
        now = time.time()
        S, G = rpcholesky(Z_train, k, block)
        prep[j, 0] = time.time() - now
        print('RPC time:', prep[j, 0])
    elif j == 1:
        # uniform Nystrom
        print('Running uniform sampling Nystrom...')
        now = time.time()
        S = np.random.choice(n_train, size=k, replace=False)
        G = kernel(Z_train[S, :], Z_train)
        C = np.linalg.cholesky(G[:, S])
        G = solve_triangular(C, G, overwrite_b=True,
                             check_finite=False, lower=True)
        prep[j, 0] = time.time() - now
        print('Uniform Nystrom time:', prep[j, 0])
    elif j == 2:
        # Greedy Nystrom
        print('Running greedy Nystrom...')
        now = time.time()
        S, G = greedy(Z_train, k)
        prep[j, 0] = time.time() - now
        print('Greedy time:', prep[j, 0])
    elif j == 3:
        # Ridge leverage scores
        print('Running ridge leverage score sampling...')
        now = time.time()
        S = recursiveNystrom(Z_train, k)
        G = kernel(Z_train[S, :], Z_train)
        C = np.linalg.cholesky(G[:, S])
        G = solve_triangular(C, G, overwrite_b=True,
                             check_finite=False, lower=True)
        prep[j, 0] = time.time() - now
        print('RLS time:', prep[j, 0])
    elif j == 4:
        # Simple RPCholesky
        print('Running RPCholesky...')
        now = time.time()
        S, G = rpc(Z_train, k)
        prep[j, 0] = time.time() - now
        print('RPC time:', prep[j, 0])
    elif j == 5:
        # Random Fourier features
        print('Running random Fourier features...')
        now = time.time()
        G = rff(Z_train, k)
        prep[j, 0] = time.time() - now
        print('RFF time:', prep[j, 0])

    # prepare preconditioner
    if j <= 4:
        print('Preparing preconditioner...')
        now = time.time()
        G = G.T
        G, C = qr(G, overwrite_a=True, mode='economic', check_finite=False)
        C, Sigma, _ = np.linalg.svd(C)
        G = G @ C
        def precondition(v):
            diag = 1 / (Sigma**2 + mu) - 1 / mu
            mv = G @ (diag * (G.T @ v)) + v / mu
            return(mv)
        prep[j, 1] = time.time() - now
        print('Preparation time:', prep[j, 1])
    else:
        def precondition(v):
            return(v)
    
    # initial iterate
    now = time.time()
    x = np.zeros(n_train)
    r = np.copy(Y_train)
    resid[j, 0] = (np.sum(r**2))**.5 / np.sum(Y_train**2)**.5
    z = precondition(r)
    p = np.copy(z)
    rz = r @ z
    times[j, 0] = time.time() - now
    print('Iteration: ', 0,)
    print('Time: ', times[j, 0])
    print('Resid: ', resid[j, 0])
    pred = ave + tall @ x
    mae[j, 0] = np.mean(np.abs(pred - Y_test))
    print('MAE:', mae[j, 0])
    rmse[j, 0] = np.mean((pred - Y_test)**2)**.5
    print('RMSE:', rmse[j, 0])

    for t in range(1, max_iters + 1):
        # update iterate
        Kp = np.copy(p)
        for i in range(blocks.size - 1):
            print('Rows ', blocks[i], ' to ', blocks[i+1])
            Kp[blocks[i]:blocks[i + 1]] = np.dot(A[blocks[i]:blocks[i + 1], :], p)
        Kp += mu * p
        alpha = rz / (p @ Kp)
        x = x + alpha * p
        r = r - alpha * Kp
        resid[j, t] = (np.sum(r**2))**.5 / np.sum(Y_train**2)**.5
        if resid[j, t] < tol:
            break
        z = precondition(r)
        beta = (r @ z) / rz
        rz = r @ z
        p = z + beta * p
        times[j, t] = time.time() - now
        print('Iteration: ', t,)
        print('Time: ', times[j, t])
        print('Resid: ', resid[j, t])
        pred = ave + tall @ x
        mae[j, t] = np.mean(np.abs(pred - Y_test))
        print('MAE:', mae[j, t])
        rmse[j, t] = np.mean((pred - Y_test)**2)**.5
        print('RMSE:', rmse[j, t])
 
# close kernel matrix
# np.savez('aspirin2.npz', prep, times, resid, mae, rmse)
# np.savez('benzene2.npz', prep, times, resid, mae, rmse)
# np.savez('ethanol2.npz', prep, times, resid, mae, rmse)
# np.savez('malonaldehyde2.npz', prep, times, resid, mae, rmse)
# np.savez('naphthalene2.npz', prep, times, resid, mae, rmse)
# np.savez('salicylic2.npz', prep, times, resid, mae, rmse)
# np.savez('toluene2.npz', prep, times, resid, mae, rmse)
np.savez('uracil2.npz', prep, times, resid, mae, rmse)
h5f_A.close()

#%%

names = np.array(['Uracil', 'Toluene', 'Salicylic acid', 'Naphthalene', 
                  'Malonaldehyde', 'Ethanol', 'Benzene', 'Aspirin'])
filenames = np.array(['uracil2.npz', 'toluene2.npz', 'salicylic2.npz', 
                      'naphthalene2.npz', 'malonaldehyde2.npz', 'ethanol2.npz', 
                      'benzene2.npz', 'aspirin2.npz'])
dims = (names.size, ) + np.load(filenames[0])['arr_0'].shape
prep = np.zeros(dims)
dims_time = (names.size, ) + np.load(filenames[0])['arr_1'].shape
times = np.zeros(dims_time)
resid = np.zeros(dims_time)
mae = np.zeros(dims_time)
rmse = np.zeros(dims_time)
for i in range(names.size):
    file = np.load(filenames[i])
    prep[i, ...] = file['arr_0']
    times[i, ...] = file['arr_1']
    resid[i, ...] = file['arr_2']
    mae[i, ...] = file['arr_3']
    rmse[i, ...] = file['arr_4']

index = np.argmax(times, axis = 2)
times = np.max(times, axis=2)
prep_times = np.sum(prep, axis=2)
total_time = prep_times + times
total_time = total_time / (60 * 60)
prep_times = prep_times / (60 * 60)
times = times / (60 * 60)

mae = np.array([mae[i, 0, index[i, 0]] for i in range(8)])
mae_ref = np.array([0.103, 0.092, 0.105, 0.113, 0.074, 0.052, 0.069, 0.127])

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(np.arange(names.size) + .33/2, mae, height=.33/1.5, color=colors[0], label='New\nmodel')
ax.barh(np.arange(names.size) - .33/2, mae_ref, height=.33/1.5, color=colors[3], label='Previous\nmodel')
ax.tick_params(width=1.5, which='both')
ax.set_xlabel('Energy error (kcal / mol)')
ax.set_xlim([0, 0.15])
ax.set_yticks(np.arange(names.size), names)
leg = ax.legend(loc='upper left', bbox_to_anchor=(1., .7), ncol=1)
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('energy_accuracy.pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(np.arange(names.size) + .3, total_time[:, 0], height = .1, color=colors[0], 
        label='Accelerated\nRPCholesky\n(OURS)')
ax.barh(np.arange(names.size) + .15, total_time[:, 3], height = .1, color=colors[1], 
        label='Ridge\nleverage\nscores')
ax.barh(np.arange(names.size), total_time[:, 1], height = .1, color=colors[2], 
        label='Uniform')
ax.barh(np.arange(names.size) - .15, total_time[:, 4], height = .1, color=colors[3], 
        label='Simple\nRPCholesky')
ax.barh(np.arange(names.size) - .3, total_time[:, 2], height = .1, color=colors[4], 
        label='Greedy')
ax.tick_params(width=1.5, which='both')
ax.set_xlabel('Hours')
ax.set_xlim([0, 6])
ax.set_yticks(np.arange(names.size), names)
leg = ax.legend(loc='upper left', bbox_to_anchor=(1., .95), ncol=1)
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('comparison_new.pdf', bbox_inches='tight')

names = np.array(['Accelerated\nRPCholesky', 'Uniform', 'Greedy', 'Ridge leverage\nscores',
                  'Simple\nRPCholesky'])
order = np.array([2, 4, 1, 3, 0])
names = names[order]
total_time = np.mean(total_time, axis=0)[order]
prep_times = np.mean(prep_times, axis=0)[order]
times = np.mean(times, axis=0)[order]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.grid(which='both')
ax.set_axisbelow(True)
ax.barh(np.arange(times.size) + .25, prep_times, height = .25/1.5, color=colors[0], 
        label='Prep\ntime')
ax.barh(np.arange(times.size), times, height = .25/1.5, color=colors[3], 
        label='PCG\ntime')
ax.barh(np.arange(times.size) - .25, total_time, height = .25/1.5, color=colors[4], 
        label='Total\ntime')
ax.tick_params(width=1.5, which='both')
ax.set_xlabel('Hours')
ax.set_xlim([0, 4.1])
ax.set_yticks(np.arange(times.size), names)
leg = ax.legend(loc='upper left', bbox_to_anchor=(1., .8), ncol=1)
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('timing.pdf', bbox_inches='tight')
