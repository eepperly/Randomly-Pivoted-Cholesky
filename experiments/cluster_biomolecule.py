#!/usr/bin/env python3

'''
Code to test clustering by Nystrom-acclerated
kernel spectral clustering for alanine
dipeptide dataset. This code was used to
 produce Figure 4 in the manuscript. Code 
written by Robert Webber and edited by
Ethan Epperly.
'''

# =====
# SETUP
# =====

import sys
sys.path.append('../')

# import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import k_means
import scipy.linalg as spl
from itertools import permutations
from unif_sample import uniform_sample
from leverage_score import recursive_rls_acc
from rpcholesky import simple_rpcholesky, greedy
from matrix import KernelMatrix

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

# Get reference clustering
kmeans_true = np.load('adp_ref.npy')
centers = 4
perms = np.array(list(permutations(np.arange(centers)))) # for later index permutations

#%%%%

##############################
## Make representative figures
##############################

sigma = .1
k = 150
rgreedy = lambda A, k: greedy(A, k, randomized_tiebreaking = True)
methods = { 'RPCholesky' : simple_rpcholesky,
            'Greedy' : rgreedy,
            'RLS' : recursive_rls_acc,
            'Uniform' : uniform_sample }
A = KernelMatrix(data, bandwidth = sigma)

for name, method in methods.items():
    print(name)
    
    lra = method(A, k)
    F = lra.get_right_factor()
    idx = lra.get_indices()
    print("Low-rank approximation done!")
    
    # dense eigensolver
    normal = F.T @ (F @ np.ones(F.shape[1]))
    normal[normal > 0] = normal[normal > 0]**-.5
    F = F * normal[np.newaxis, :]
    vals, vecs = np.linalg.svd(F, full_matrices=False)[1:3]
    vecs = vecs.T * normal[:, np.newaxis]

    # k-means clustering
    centers = 4
    v_min = 3
    points = vecs[:, 1:v_min]
    kmeans = k_means(points, centers)[1]

    accuracy = np.zeros(perms.shape[0])
    for l in range(perms.shape[0]):
            kmeans_flipped = np.zeros(kmeans.size)
            for j in range(centers):
                kmeans_flipped[kmeans == j] = perms[l, j]
            accuracy[l] = np.mean(np.equal(kmeans_flipped, kmeans_true))
    print('accuracy:', accuracy.max())
    kmeans_flipped = np.copy(kmeans)
    perm = perms[np.argmax(accuracy), :]
    for j in range(centers):
        kmeans_flipped[kmeans == j] = perm[j]
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
    fig.savefig('figs/{}_clustering.png'.format(name), bbox_inches = 'tight')

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
    fig.savefig('figs/{}_clustering_2.png'.format(name), bbox_inches = 'tight')

###########################################
## Long slow process to figure out accuracy
###########################################

k_max = 200
iterations = 1000
accuracies = []
k_range = np.arange(10, k_max + 1, 10)

for name, method in methods.items():
    print(name)
    method_accuracy = np.zeros((iterations, len(k_range)))
    for j in range(iterations):
        print("Iteration {}".format(j))
        for k_idx, k in zip(range(len(k_range)), k_range):

            # dense eigensolver
            lra = method(A, k)
            F = lra.get_right_factor()
            normal = F.T @ (F @ np.ones(F.shape[1]))
            normal[normal > 0] = normal[normal > 0]**-.5
            F = F * normal[np.newaxis, :]
            vals, vecs = np.linalg.svd(F, full_matrices=False)[1:3]
            vecs = vecs.T * normal[:, np.newaxis]

            # k-means clustering
            centers = 4
            v_min = 3
            points = vecs[:, 1:v_min]
            kmeans = k_means(points, centers)[1]
            accuracy = np.zeros(perms.shape[0])
            for l in range(perms.shape[0]):
                    kmeans_flipped = np.zeros(kmeans.size)
                    for p in range(centers):
                        kmeans_flipped[kmeans == p] = perms[l, p]
                    accuracy[l] = np.mean(kmeans_flipped == kmeans_true)
            print('accuracy: ', accuracy.max())
            method_accuracy[j, k_idx] = accuracy.max()

        # print output
        print('{}: '.format(name), j, np.mean(method_accuracy[:j+1, :], axis = 0))
        
    accuracies.append(method_accuracy)

####################################
## Consolidate the different methods
####################################

fig, ax = plt.subplots(figsize = (9, 5))
linestyles = ["-.","*-","--",":"]
for i, name, accuracy in zip(range(len(methods)), methods.keys(), accuracies):
    ax.semilogy(k_range, 1 - np.mean(accuracy, axis=0), linestyles[i],
                color=colors[i], label=name)
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

np.savez('data/results.npz', np.array(accuracies))
