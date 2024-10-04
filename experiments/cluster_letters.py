#!/usr/bin/env python3

'''
Code to test clustering by Nystrom-acclerated
kernel spectral clustering for clustering
letters. This code was used to produce Figure 4
in v1-v3 of the manuscript together with
'matlab_plotting/make_clustering_plots.m'
'''

import sys
sys.path.append('../')

# import libraries
from dpp_lra import dpp_cubic
from unif_sample import uniform_sample
from leverage_score import recursive_rls_acc
from rpcholesky import simple_rpcholesky, greedy

import numpy as np
from scipy.io import savemat
from matrix import KernelMatrix
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from cluster_letters_plot import get_image_data

_, _, X, K = get_image_data()
true_labels = np.zeros(X.shape[0], dtype=int)

true_labels[X[:,0] > 0.942] = 1
true_labels[X[:,1] < 2.73973 * X[:,0] - 4.43836] = 2
    
ks = range(20,220,20)
num_trials = 100

methods = { 'DPP' : dpp_cubic,
            'Greedy' : greedy,
            'RLS' : recursive_rls_acc,
            'Uniform' : uniform_sample,
            'RPCholesky' : simple_rpcholesky }

for name, method in methods.items():
    accuracies = np.zeros((len(ks), num_trials))
    print(name)
    if method == 'DPP':
        A = K[:,:]
    for k, idx in zip(ks, range(len(ks))):
        print(" ", k)
        for i in range(num_trials):
            try:
                if method == 'DPP':
                    lra = method(A, k)
                else:
                    lra = method(K, k)
                row_sums = (lra @ np.ones((X.shape[0],1))).flatten()
                evecs = lra.scale(row_sums ** -0.5).eigenvalue_decomposition().V
                evecs = (row_sums[:,np.newaxis] ** -0.5) * evecs
                points = evecs[:,0:3]
                kmeans = KMeans(n_clusters=3, random_state=0).fit(points)

                accuracies[idx, i] = normalized_mutual_info_score(true_labels, kmeans.labels_)
            except np.linalg.LinAlgError:
                accuracies[idx, i] = np.NaN
            print("   ", accuracies[idx,i])
        print("  Mean:", np.mean(accuracies[idx,:]))

    output = {"accuracies" : accuracies}
    savemat("data/{}_accuracies.mat".format(name),output)

    print()

