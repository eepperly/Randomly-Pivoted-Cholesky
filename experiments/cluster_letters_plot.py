#!/usr/bin/env python3

'''
Code to show clustering for produced by Nystrom-
acclerated kernel spectral clustering. To use,
set 'method' to the desired Nystrom method and
'k' to the desired number of pivots. This code
was used to produce Figure 4 in v1-v3 of the
manuscript
'''

import sys
sys.path.append('../')

# import libraries
from dpp_lra import dpp_cubic
from unif_sample import uniform_sample
from leverage_score import recursive_rls_acc
from rpcholesky import simple_rpcholesky, greedy

import numpy as np
from pdf2image import convert_from_path
from sklearn.metrics import normalized_mutual_info_score
from matrix import KernelMatrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_image_data():
    # Load data 
    images = convert_from_path('RPC.pdf')
    image = np.array(images[0])[..., 0]
    y, x = np.where(image == 0)
    y = y.max() - y
    x = x - x.min()
    x = x / y.max()
    y = y / y.max()
    X = np.zeros((len(x), 2))
    X[:,0] = x
    X[:,1] = y
    K = KernelMatrix(X, bandwidth = 0.05)

    return x, y, X, K

if __name__ == "__main__":
    x, y, X, K = get_image_data()
    true_labels = np.zeros(X.shape[0], dtype=int)

    true_labels[X[:,0] > 0.942] = 1
    true_labels[X[:,1] < 2.73973 * X[:,0] - 4.43836] = 2
    
    method = simple_rpcholesky
    k = 140

    lra = method(K, k)
    row_sums = (lra @ np.ones((len(x),1))).flatten()
    evecs = lra.scale(row_sums ** -0.5).eigenvalue_decomposition().V

    evecs = (row_sums[:,np.newaxis] ** -0.5) * evecs
    points = evecs[:,0:3]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(points)
    labels = kmeans.labels_

    # Fix label order
    idx = np.argmin(x)
    first_label = labels[idx]

    idx = np.argmin(x[labels != first_label])
    second_label = labels[labels != first_label][idx]

    all_labels = [0,1,2]
    all_labels.remove(first_label)
    all_labels.remove(second_label)
    third_label = all_labels[0]
    labels[labels == first_label] = 3
    labels[labels == second_label] = 4
    labels[labels == third_label] = 5
    labels[labels == 3] = 0
    labels[labels == 4] = 1
    labels[labels == 5] = 2

    colors = np.array(['#332288', '#44AA99', '#88CCEE'])

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(x, y, c = colors[kmeans.labels_])
    ax.set_aspect('equal')
    plt.show()
    plt.savefig("figs/clustered.png")

    print(normalized_mutual_info_score(true_labels, kmeans.labels_))
