#!/usr/bin/env python

'''
Code to evaluate the error of different Nystrom methods
on the largest molecules in the QM9 dataset, from which
the claim in the manuscrip that uniform sampling has an
18x higher SMAPE than RPCholesky on the largest molecules
in the QM9 dataset was based. To use this code, the 
QM9 dataset must first be stored in 'data/homo.mat'
as is done by qm9_krr.py
'''

import scipy.io
from scipy.io import savemat
import numpy as np
from KRR_Nystrom import KRR_Nystrom
import rp_cholesky, leverage_score, unif_sample

data = scipy.io.loadmat('data/homo.mat')
X = data['X']
feature = X
target = data['Y'].flatten()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
feature = scaler.fit_transform(feature)
n,d = np.shape(feature)

num_train = 100000
num_test = n - num_train

train_sample = feature[:num_train]
train_sample_target = target[:num_train]
test_sample = feature[num_train:num_train+num_test]
test_sample_target = target[num_train:num_train+num_test]

k = 1000
lamb = 1.0e-8
sigma = 5120.0
solve_method = 'Direct'

num_trials = 100

rpcholesky_largest_atom_error = 0.0
uniform_largest_atom_error = 0.0

methods = { 'Uniform' : unif_sample.uniform_sample,
            'RLS' : leverage_score.recursive_rls_acc,
            'RPCholesky' : rp_cholesky.rp_cholesky,
            'Greedy' : rp_cholesky.greedy }

all_errors = { 'RLS' : [], 'Uniform' : [], 'RPCholesky' : [], 'Greedy' : [] }
all_smapes = { 'RLS' : [], 'Uniform' : [], 'RPCholesky' : [], 'Greedy' : [] }

for trial in range(num_trials):
    print("Running trial", trial+1)
    
    for method_name, method in methods.items():
        while True:
            try:
                model = KRR_Nystrom(kernel = "laplace", bandwidth = sigma)
                model.fit_Nystrom(train_sample, train_sample_target, lamb = lamb, sample_num = k, sample_method = method, solve_method = solve_method)
                preds  = model.predict_Nystrom(test_sample)
                smapes = 2 * np.abs(test_sample_target - preds) / (np.abs(test_sample_target) + np.abs(preds))
                errors = np.abs(test_sample_target - preds)
                break
            except np.linalg.LinAlgError:
                pass

        nonzeros = np.count_nonzero(X[num_train:num_train+num_test], axis=1)
        all_smapes[method_name].append(np.mean(smapes[nonzeros == np.amax(nonzeros)]))
        all_errors[method_name].append(np.mean(errors[nonzeros == np.amax(nonzeros)]))

        print("\t", method_name, np.mean(all_smapes[method_name]), np.mean(all_errors[method_name]))
