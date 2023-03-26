#!/usr/bin/env python3

'''
Code to test clustering by Nystrom-acclerated
kernel spectral clustering for alanine
dipeptide dataset. This code was used to
 produce Figure 4 in the manuscript. Code 
written by Robert Webber.
'''

# =====
# SETUP
# =====

# import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import k_means
import scipy.linalg as spl
from itertools import permutations

# pretty plots
font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 1.5
colors = np.array(['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'])
colors2 = np.array(['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'])[np.array([0, 2, 1, 3, 4])]

#%%%%

##################################
## Get data and obtain kmeans_true
##################################

# Get data
npzfile = np.load('alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
angles = np.array(npzfile['arr_0'], dtype = float)
npzfile = np.load('alanine-dipeptide-3x250ns-heavy-atom-positions.npz')
data = np.array(npzfile['arr_0'], dtype = float)

# RPC
def rpc2(data, sigma, k):
    N = data.shape[0]
    diags = np.ones(N)
    F = np.zeros((k,N))
    arr_idx = []
    
    for i in range(k):
        idx = np.random.choice(N, p = diags / np.sum(diags))
        arr_idx.append(idx)
        diff = data[idx, :] - data
        col = np.sum(diff**2, axis = 1)
        col = np.exp(-.5 * col / sigma ** 2)
        F[i,:] = (col - F[:i,idx].T @ F[:i,:]) / np.sqrt(diags[idx])
        diags -= F[i,:]**2
        diags = diags.clip(min = 0)

    return(F, arr_idx)

# Apply clustering
sigma = .1
k = 1000
rpc_mat, idx = rpc2(data, sigma, k)

# dense eigensolver
normal = rpc_mat.T @ (rpc_mat @ np.ones(rpc_mat.shape[1]))
normal[normal > 0] = normal[normal > 0]**-.5
rpc_mat = rpc_mat * normal[np.newaxis, :]
vals, vecs = np.linalg.svd(rpc_mat, full_matrices=False)[1:3]
vecs = vecs.T * normal[:, np.newaxis]

# k-means clustering
centers = 4
v_min = 3
points = vecs[:, 1:v_min]
kmeans = k_means(points, centers)[1]
order = (kmeans[:, np.newaxis] == np.arange(centers)[np.newaxis, :])
order = np.sum(angles[:, 0, np.newaxis] * order, axis = 0) / np.sum(order, axis = 0)
order = np.argsort(order)
kmeans = (kmeans[:, np.newaxis] == order[np.newaxis, :]) @ np.arange(centers)
# kmeans_true = np.copy(kmeans)
# np.save('adp_ref.npy', kmeans_true)
kmeans_true = np.load('adp_ref.npy')
perms = np.array(list(permutations(np.arange(centers)))) # for later index permutations

# plot clustering results
fig = plt.figure(figsize = (7, 5))
ax = fig.gca()
ax.scatter(angles[:, 0], angles[:, 1], s=.02, c=colors2[kmeans_true], alpha=1)
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect('equal')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlabel('$\phi$ angle')
ax.set_ylabel('$\psi$ angle')
ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.tick_params(width=1.5, which='both')
fig.tight_layout()
#fig.savefig('clustering_ref.png', bbox_inches = 'tight')

#%%%%

####################################
## RPC: Make a representative figure
####################################

# Apply clustering
sigma = .1
k = 150
rpc_mat, idx = rpc2(data, sigma, k)

# dense eigensolver
normal = rpc_mat.T @ (rpc_mat @ np.ones(rpc_mat.shape[1]))
normal[normal > 0] = normal[normal > 0]**-.5
rpc_mat = rpc_mat * normal[np.newaxis, :]
vals, vecs = np.linalg.svd(rpc_mat, full_matrices=False)[1:3]
vecs = vecs.T * normal[:, np.newaxis]

# k-means clustering
centers = 4
v_min = 3
points = vecs[:, 1:v_min]
kmeans = k_means(points, centers)[1]


accuracy = np.zeros(perms.shape[0])
for l in range(perms.shape[0]):
        kmeans_flipped = np.zeros(kmeans.size)
        for k in range(centers):
            kmeans_flipped[kmeans == k] = perms[l, k]
        accuracy[l] = np.mean(kmeans_flipped == kmeans_true)
print('accuracy: ', accuracy.max())
kmeans_flipped = np.copy(kmeans)
perm = perms[np.argmax(accuracy), :]
for k in range(centers):
    kmeans_flipped[kmeans == k] = perm[k]
kmeans = kmeans_flipped

# plot clustering results
fig = plt.figure(figsize = (6.5, 4.5))
ax = fig.gca()
ax.scatter(angles[:, 0], angles[:, 1], s = .02, c = colors2[kmeans])
ax.scatter(angles[idx, 0], angles[idx, 1], s = 5, c = 'black')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect('equal')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlabel('$\phi$ angle')
ax.set_ylabel('$\psi$ angle')
ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.tick_params(width=1.5, which='both')
fig.tight_layout()
fig.savefig('figs/rpc_clustering.png', bbox_inches = 'tight')

# plot clustering results
fig = plt.figure(figsize = (6.5, 4.5))
ax = fig.gca()
ax.scatter(angles[:, 0], angles[:, 1], s = .02, c = colors2[kmeans])
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect('equal')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlabel('$\phi$ angle')
ax.set_ylabel('$\psi$ angle')
ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.tick_params(width=1.5, which='both')
fig.tight_layout()
fig.savefig('figs/rpc_clustering_2.png', bbox_inches = 'tight')

#%%%%

################################################
## RPC: Long slow process to figure out accuracy
################################################

# Apply RPC
k_max = 200
iterations = 1000
rpc_accuracy = np.zeros((iterations, k_max + 1))
rpc_range = np.arange(10, k_max + 1, 10)

for j in range(iterations):
    rpc_ref = rpc2(data, sigma, k_max)[0]
    for i in rpc_range:
        print(i)

        # dense eigensolver
        rpc_mat = rpc_ref[:i, :]
        normal = rpc_mat.T @ (rpc_mat @ np.ones(rpc_mat.shape[1]))
        normal[normal > 0] = normal[normal > 0]**-.5
        rpc_mat = rpc_mat * normal[np.newaxis, :]
        vals, vecs = np.linalg.svd(rpc_mat, full_matrices=False)[1:3]
        vecs = vecs.T * normal[:, np.newaxis]
        
        # k-means clustering
        centers = 4
        v_min = 3
        points = vecs[:, 1:v_min]
        kmeans = k_means(points, centers)[1]
        accuracy = np.zeros(perms.shape[0])
        for l in range(perms.shape[0]):
                kmeans_flipped = np.zeros(kmeans.size)
                for k in range(centers):
                    kmeans_flipped[kmeans == k] = perms[l, k]
                accuracy[l] = np.mean(kmeans_flipped == kmeans_true)
        print('accuracy: ', accuracy.max())
        rpc_accuracy[j, i] = accuracy.max()
        
    # print output
    print(('RPC: ', j, np.mean(rpc_accuracy[:j+1, rpc_range], axis = 0)))

#%%%%

#######################################
## Greedy: Make a representative figure
#######################################

# Greedy
def greedy2(data, sigma, k):
    N = data.shape[0]
    diags = np.ones(N)
    F = np.zeros((k,N))
    arr_idx = []
    
    for i in range(k):
        idx = np.random.choice(np.where(diags == np.max(diags))[0])
        arr_idx.append(idx)
        diff = data[idx, :] - data
        col = np.sum(diff**2, axis = 1)
        col = np.exp(-.5 * col / sigma ** 2)
        F[i,:] = (col - F[:i,idx].T @ F[:i,:]) / np.sqrt(diags[idx])
        diags -= F[i,:]**2
        diags = diags.clip(min = 0)

    return(F, arr_idx)

# Apply clustering
sigma = .1
k = 150
greedy_mat, idx = greedy2(data, sigma, k)

# dense eigensolver
normal = greedy_mat.T @ (greedy_mat @ np.ones(greedy_mat.shape[1]))
normal[normal > 0] = normal[normal > 0]**-.5
greedy_mat = greedy_mat * normal[np.newaxis, :]
vecs = np.linalg.svd(greedy_mat, full_matrices=False)[2]
vecs = vecs.T * normal[:, np.newaxis]

# k-means clustering
centers = 4
v_min = 3
points = vecs[:, 1:v_min]
kmeans = k_means(points, centers)[1]
accuracy = np.zeros(perms.shape[0])
for l in range(perms.shape[0]):
        kmeans_flipped = np.zeros(kmeans.size)
        for k in range(centers):
            kmeans_flipped[kmeans == k] = perms[l, k]
        accuracy[l] = np.mean(kmeans_flipped == kmeans_true)
print('accuracy: ', accuracy.max())
kmeans_flipped = np.copy(kmeans)
perm = perms[np.argmax(accuracy), :]
for k in range(centers):
    kmeans_flipped[kmeans == k] = perm[k]
kmeans = kmeans_flipped

# plot clustering results
fig = plt.figure(figsize = (6.5, 4.5))
ax = fig.gca()
ax.scatter(angles[:, 0], angles[:, 1], s=.02, c=colors2[kmeans])
ax.scatter(angles[idx, 0], angles[idx, 1], s = 5, c = 'black')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect('equal')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlabel('$\phi$ angle')
ax.set_ylabel('$\psi$ angle')
ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.tick_params(width=1.5, which='both')
fig.tight_layout()
fig.savefig('figs/greedy_clustering.png', bbox_inches = 'tight')

# plot clustering results
fig = plt.figure(figsize = (6.5, 4.5))
ax = fig.gca()
ax.scatter(angles[:, 0], angles[:, 1], s=.02, c=colors2[kmeans])
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect('equal')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlabel('$\phi$ angle')
ax.set_ylabel('$\psi$ angle')
ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.tick_params(width=1.5, which='both')
fig.tight_layout()
fig.savefig('figs/greedy_clustering_2.png', bbox_inches = 'tight')

#%%%%

###################################################
## Greedy: Long slow process to figure out accuracy
###################################################

# Apply greedy
k_max = 200
iterations = 1000
greedy_accuracy = np.zeros((iterations, k_max + 1))
greedy_range = np.arange(10, k_max + 1, 10)

for j in range(iterations):
    greedy_ref = greedy2(data, sigma, k_max)[0]
    for i in greedy_range:
        print(i)

        # dense eigensolver
        greedy_mat = greedy_ref[:i, :]
        normal = greedy_mat.T @ (greedy_mat @ np.ones(greedy_mat.shape[1]))
        normal[normal > 0] = normal[normal > 0]**-.5
        greedy_mat = greedy_mat * normal[np.newaxis, :]
        vals, vecs = np.linalg.svd(greedy_mat, full_matrices=False)[1:3]
        vecs = vecs.T * normal[:, np.newaxis]
        
        # k-means clustering
        centers = 4
        v_min = 3
        points = vecs[:, 1:v_min]
        kmeans = k_means(points, centers)[1]
        accuracy = np.zeros(perms.shape[0])
        for l in range(perms.shape[0]):
                kmeans_flipped = np.zeros(kmeans.size)
                for k in range(centers):
                    kmeans_flipped[kmeans == k] = perms[l, k]
                accuracy[l] = np.mean(kmeans_flipped == kmeans_true)
        print('accuracy: ', accuracy.max())
        greedy_accuracy[j, i] = accuracy.max()
        
    # print output
    print(('Greedy: ', j, np.mean(greedy_accuracy[:j+1, greedy_range], axis = 0)))

#%%%%

########################################
## Uniform: Make a representative figure
########################################

# Uniform
def unif2(data, sigma, k):
    N = data.shape[0]
    diags = np.ones(N)
    F = np.zeros((k,N))
    arr_idx = np.random.choice(N, size = k, replace=False)
    
    for i in range(k):
        diff = data[arr_idx[i], :] - data
        col = np.sum(diff**2, axis = 1)
        col = np.exp(-.5 * col / sigma ** 2)
        F[i,:] = (col - F[:i,arr_idx[i]].T @ F[:i,:]) / np.sqrt(diags[arr_idx[i]])
        diags -= F[i,:]**2
        diags = diags.clip(min = 0)

    return(F, arr_idx)

# Apply clustering
sigma = .1
k = 150
unif_mat, idx = unif2(data, sigma, k)

# dense eigensolver
normal = unif_mat.T @ (unif_mat @ np.ones(unif_mat.shape[1]))
normal[normal > 0] = normal[normal > 0]**-.5
unif_mat = unif_mat * normal[np.newaxis, :]
vecs = np.linalg.svd(unif_mat, full_matrices=False)[2]
vecs = vecs.T * normal[:, np.newaxis]

# k-means clustering
centers = 4
v_min = 3
points = vecs[:, 1:v_min]
kmeans = k_means(points, centers)[1]
accuracy = np.zeros(perms.shape[0])
for l in range(perms.shape[0]):
        kmeans_flipped = np.zeros(kmeans.size)
        for k in range(centers):
            kmeans_flipped[kmeans == k] = perms[l, k]
        accuracy[l] = np.mean(kmeans_flipped == kmeans_true)
print('accuracy: ', accuracy.max())
kmeans_flipped = np.copy(kmeans)
perm = perms[np.argmax(accuracy), :]
for k in range(centers):
    kmeans_flipped[kmeans == k] = perm[k]
kmeans = kmeans_flipped

# plot clustering results
fig = plt.figure(figsize = (6.5, 4.5))
ax = fig.gca()
ax.scatter(angles[:, 0], angles[:, 1], s = .02, c = colors2[kmeans])
ax.scatter(angles[idx, 0], angles[idx, 1], s = 5, c = 'black')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect('equal')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlabel('$\phi$ angle')
ax.set_ylabel('$\psi$ angle')
ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.tick_params(width=1.5, which='both')
fig.tight_layout()
fig.savefig('figs/unif_clustering.png', bbox_inches = 'tight')

# plot clustering results
fig = plt.figure(figsize = (6.5, 4.5))
ax = fig.gca()
ax.scatter(angles[:, 0], angles[:, 1], s = .02, c = colors2[kmeans])
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect('equal')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlabel('$\phi$ angle')
ax.set_ylabel('$\psi$ angle')
ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.tick_params(width=1.5, which='both')
fig.tight_layout()
fig.savefig('figs/unif_clustering_2.png', bbox_inches = 'tight')

#%%%%

####################################################
## Uniform: Long slow process to figure out accuracy
####################################################

# Apply uniform
k_max = 200
iterations = 1000
unif_accuracy = np.zeros((iterations, k_max + 1))
unif_range = np.arange(10, k_max + 1, 10)

for j in range(iterations):
    unif_ref = unif2(data, sigma, k_max)[0]
    for i in unif_range:
        print(i)

        # dense eigensolver
        unif_mat = unif_ref[:i, :]
        normal = unif_mat.T @ (unif_mat @ np.ones(unif_mat.shape[1]))
        normal[normal > 0] = normal[normal > 0]**-.5
        unif_mat = unif_mat * normal[np.newaxis, :]
        vals, vecs = np.linalg.svd(unif_mat, full_matrices=False)[1:3]
        vecs = vecs.T * normal[:, np.newaxis]
        
        # k-means clustering
        centers = 4
        v_min = 3
        points = vecs[:, 1:v_min]
        kmeans = k_means(points, centers)[1]
        accuracy = np.zeros(perms.shape[0])
        for l in range(perms.shape[0]):
                kmeans_flipped = np.zeros(kmeans.size)
                for k in range(centers):
                    kmeans_flipped[kmeans == k] = perms[l, k]
                accuracy[l] = np.mean(kmeans_flipped == kmeans_true)
        print('accuracy: ', accuracy.max())
        unif_accuracy[j, i] = accuracy.max()

    # print output
    print(('Uniform: ', j, np.mean(unif_accuracy[:j+1, unif_range], axis = 0)))

#%%%%

#############################################
## RLS Sampling: Make a representative figure
#############################################

sigma = .1

def gauss(X: np.ndarray, Y: np.ndarray=None, sigma=sigma):
    # todo make this implementation more python like!

    if Y is None:
        Ksub = np.ones((X.shape[0], 1))
    else:
        nsq_rows = np.sum(X ** 2, axis=1, keepdims=True)
        nsq_cols = np.sum(Y ** 2, axis=1, keepdims=True)
        Ksub = nsq_rows - np.matmul(X, Y.T * 2)
        Ksub = nsq_cols.T + Ksub
        Ksub = np.exp(-.5 * Ksub / sigma**2)

    return Ksub

'''
This function is taken from https://github.com/cnmusco/recursive-nystrom, which
is licensed under the MIT License:


MIT License

Copyright (c) 2017 Christopher Musco and Cameron Musco

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

def recursiveNystrom(X, n_components: int, kernel_func=gauss, accelerated_flag=False, random_state=None, lmbda_0=0, return_leverage_score=False, **kwargs):
    '''
    :param X:
    :param n_components:
    :param kernel_func:
    :param accelerated_flag:
    :param random_state:
    :return:
    '''
    rng = np.random.RandomState(random_state)

    n_oversample = np.log(n_components)
    k = np.ceil(n_components / (4 * n_oversample)).astype(int)
    n_levels = np.ceil(np.log(X.shape[0] / n_components) / np.log(2)).astype(int)
    perm = rng.permutation(X.shape[0])

    # set up sizes for recursive levels
    size_list = [X.shape[0]]
    for l in range(1, n_levels+1):
        size_list += [np.ceil(size_list[l - 1] / 2).astype(int)]

    # indices of poitns selected at previous level of recursion
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
                    - np.sum(spl.eigvalsh(SKS * weights[:,None] * weights[None,:], eigvals=(SKS.shape[0]-k, SKS.shape[0]-1))))/k
        lmbda = np.maximum(lmbda_0*SKS.shape[0], lmbda)
        if lmbda == lmbda_0*SKS.shape[0]:
            print("Set lambda to %d." % lmbda)
        #lmbda = np.minimum(lmbda, 5)
            # lmbda = spl.eigvalsh(SKS * weights * weights.T, eigvals=(0, SKS.shape[0]-k-1)).sum()/k
            # calculate the n-k smallest eigenvalues

        # compute and sample by lambda ridge leverage scores
        R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
        R = np.matmul(KS, R)
        #R = np.linalg.lstsq((SKS + np.diag(lmbda * weights ** (-2))).T,KS.T)[0].T
        if l != 0:
            # max(0, . ) helps avoid numerical issues, unnecessary in theory
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

# RLS
def rls2(data, sigma, k):
    N = data.shape[0]
    diags = np.ones(N)
    F = np.zeros((k,N))
    arr_idx = recursiveNystrom(data, k)
    
    for i in range(k):
        diff = data[arr_idx[i], :] - data
        col = np.sum(diff**2, axis = 1)
        col = np.exp(-.5 * col / sigma ** 2)
        F[i,:] = (col - F[:i,arr_idx[i]].T @ F[:i,:]) / np.sqrt(diags[arr_idx[i]])
        diags -= F[i,:]**2
        diags = diags.clip(min = 0)

    return(F, arr_idx)

# Apply clustering
k = 150
sigma = .1
rls_mat, idx = rls2(data, sigma, k)

# dense eigensolver
normal = rls_mat.T @ (rls_mat @ np.ones(rls_mat.shape[1]))
normal[normal > 0] = normal[normal > 0]**-.5
rls_mat = rls_mat * normal[np.newaxis, :]
vecs = np.linalg.svd(rls_mat, full_matrices=False)[2]
vecs = vecs.T * normal[:, np.newaxis]

# k-means clustering
centers = 4
v_min = 3
points = vecs[:, 1:v_min]
kmeans = k_means(points, centers)[1]
accuracy = np.zeros(perms.shape[0])
for l in range(perms.shape[0]):
        kmeans_flipped = np.zeros(kmeans.size)
        for k in range(centers):
            kmeans_flipped[kmeans == k] = perms[l, k]
        accuracy[l] = np.mean(kmeans_flipped == kmeans_true)
print('accuracy: ', accuracy.max())
kmeans_flipped = np.copy(kmeans)
perm = perms[np.argmax(accuracy), :]
for k in range(centers):
    kmeans_flipped[kmeans == k] = perm[k]
kmeans = kmeans_flipped

# plot clustering results
fig = plt.figure(figsize = (6.5, 4.5))
ax = fig.gca()
ax.scatter(angles[:, 0], angles[:, 1], s=.02, c=colors2[kmeans])
ax.scatter(angles[idx, 0], angles[idx, 1], s = 5, c = 'black')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect('equal')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlabel('$\phi$ angle')
ax.set_ylabel('$\psi$ angle')
ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.tick_params(width=1.5, which='both')
fig.tight_layout()
fig.savefig('figs/rls_clustering.png', bbox_inches = 'tight')

# plot clustering results
fig = plt.figure(figsize = (6.5, 4.5))
ax = fig.gca()
ax.scatter(angles[:, 0], angles[:, 1], s=.02, c=colors2[kmeans])
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect('equal')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xlabel('$\phi$ angle')
ax.set_ylabel('$\psi$ angle')
ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
ax.tick_params(width=1.5, which='both')
fig.tight_layout()
fig.savefig('figs/rls_clustering_2.png', bbox_inches = 'tight')

#%%%%

################################################
## RLS: Long slow process to figure out accuracy
################################################

# Apply RLS
k_max = 200
iterations = 1000
rls_accuracy = np.zeros((iterations, k_max + 1))
rls_range = np.arange(10, k_max + 1, 10)

for j in range(iterations):
    for i in rls_range:

        # dense eigensolver
        rls_mat = rls2(data, sigma, i)[0]
        normal = rls_mat.T @ (rls_mat @ np.ones(rls_mat.shape[1]))
        normal[normal > 0] = normal[normal > 0]**-.5
        rls_mat = rls_mat * normal[np.newaxis, :]
        vals, vecs = np.linalg.svd(rls_mat, full_matrices=False)[1:3]
        vecs = vecs.T * normal[:, np.newaxis]
        
        # k-means clustering
        centers = 4
        v_min = 3
        points = vecs[:, 1:v_min]
        kmeans = k_means(points, centers)[1]
        accuracy = np.zeros(perms.size)
        for l in range(perms.shape[0]):
                kmeans_flipped = np.zeros(kmeans.size)
                for k in range(centers):
                    kmeans_flipped[kmeans == k] = perms[l, k]
                accuracy[l] = np.mean(kmeans_flipped == kmeans_true)
        print('accuracy: ', accuracy.max())
        rls_accuracy[j, i] = accuracy.max()

    # print output
    print(('RLS: ', j, np.mean(rls_accuracy[:j+1, rls_range], axis = 0)))

#%%%%%

####################################
## Consolidate the different methods
####################################

fig, ax = plt.subplots(figsize = (9, 5))
ax.semilogy(rpc_range, 1 - np.mean(rpc_accuracy[:, rpc_range], axis=0), '-.',
        color=colors[0], label='RPC')
ax.semilogy(greedy_range, 1 - np.mean(greedy_accuracy[:, greedy_range], axis=0), '*-',
        color=colors[1], label='Greedy')
ax.semilogy(rls_range, 1 - np.mean(rls_accuracy[:, rls_range], axis=0), '--',
        color=colors[2], label='RLS')
ax.semilogy(unif_range, 1 - np.mean(unif_accuracy[:, unif_range], axis=0), ':',
        color=colors[3], label='Uniform')
ax.set_xlim([0, 200])
ax.set_ylim([1e-3, .5])
ax.tick_params(width=1.5, which='both')
ax.set_xlabel('Approximation rank $k$')
ax.set_ylabel('Clustering error')
leg = ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1.2), ncol=4)
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((0, 0, 0, 0))
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('White')
fig.savefig('figs/clustering_acc.png', dpi=200, bbox_inches='tight')

#%%%%

####################################
## Save the data
####################################

np.savez('data/results.npz', rpc_accuracy, greedy_accuracy, unif_accuracy, rls_accuracy)
