#!/usr/bin/env python3

'''
Code to perform KRR on the QM9 dataset using different
Nystrom methods, using 100k randomly selected molecules
as training points. The l1 Laplace kernel is used with
a bandwidth 5120, and the regularization parameter is
1e-8; both were chosen using cross-validation. This code
was used to produce Figure 3 in the manuscript together
with 'matlab_plotting/make_krr_plots.m'
'''

import sys
sys.path.append('../')

import os
from scipy.io import savemat, loadmat
import numpy as np

NUCLEAR_CHARGES = {"H" : 1, "C" : 6, "N" : 7, "O" : 8, "F" : 9}

def coulomb_matrix(charges, coords, size):
    '''
    Coulomb matrix of a molecule, padded with zeros to 'size' atoms,
    with rows and columns sorted by decreasing row norm. The lower triangle
    is returned as a vector.
    '''
    n = len(charges)
    dists = np.linalg.norm(coords[:,np.newaxis,:] - coords[np.newaxis,:,:], axis=-1)
    np.fill_diagonal(dists, 1.0)
    M = np.outer(charges, charges) / dists
    np.fill_diagonal(M, 0.5 * charges ** 2.4)
    M_padded = np.zeros((size, size))
    M_padded[:n,:n] = M
    order = np.argsort(-np.linalg.norm(M_padded, axis=1), kind="stable")
    M_padded = M_padded[np.ix_(order, order)]
    return M_padded[np.tril_indices(size)]

def read_xyz(filename):
    with open(filename) as myfile:
        lines = myfile.readlines()
    n = int(lines[0])
    elements = []
    coords = np.zeros((n,3))
    for i in range(n):
        fields = lines[2+i].split()
        elements.append(fields[0])
        coords[i,:] = [float(x) for x in fields[1:4]]
    charges = np.array([NUCLEAR_CHARGES[e] for e in elements], dtype=float)
    return charges, coords, lines[1]

def get_molecules(directory = "molecules/", max_atoms = 29, max_mols = np.inf, output_index = 7):
    representations = []
    energies = []
    for f in sorted(os.listdir(directory)):
        if len(representations) >= max_mols:
            break
        if not f.endswith(".xyz"):
            continue

        try:
            charges, coords, properties = read_xyz(os.path.join(directory, f))
            representations.append(coulomb_matrix(charges, coords, max_atoms))
            energies.append(float(properties.split()[output_index]) * 27.2114) # Hartrees to eV
        except (ValueError, KeyError):
            # A few files use Mathematica-style exponents (e.g., 1.0*^-6) that cannot be parsed
            if len(representations) > len(energies):
                representations.pop()
    
    c = list(zip(representations, energies))
    np.random.shuffle(c)
    representations, energies = zip(*c)

    X = np.array(representations)
    Y = np.array(energies).reshape((X.shape[0],1))

    return X, Y 
    
if __name__ == "__main__":
    if not os.path.isfile("data/homo.mat"):
        X, Y = get_molecules()
        data = { "X" : X, "Y" : Y }
        savemat("data/homo.mat", data)
    else:
        data = loadmat("data/homo.mat")
        
    feature = data['X']
    target = data['Y'].flatten()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)
    n,d = np.shape(feature)
    
    num_train = 100000
    num_test = n - num_train
    ks = range(200, 1200, 200)

    train_sample = feature[:num_train]
    train_sample_target = target[:num_train]
    test_sample = feature[num_train:num_train+num_test]
    test_sample_target = target[num_train:num_train+num_test]

    def mean_squared_error(true, pred):
        return np.mean((true - pred)**2)
    def mean_average_error(true, pred):
        return np.mean(np.abs(true - pred))
    def SMAPE(true,pred):
        return np.mean(abs(true - pred)/((abs(true)+abs(pred))/2))

    from KRR_Nystrom import KRR_Nystrom
    import rpcholesky
    import leverage_score
    import unif_sample
    import matplotlib.pyplot as plt
    import time
    from functools import partial

    methods = { 'Greedy' : rpcholesky.greedy,
                'Uniform' : unif_sample.uniform_sample,
                'RPCholesky' : rpcholesky.simple_rpcholesky,
                'RLS' : leverage_score.recursive_rls_acc,
                'block50RPCholesky' : partial(rpcholesky.block_rpcholesky,b=50) }

    num_trials = 100
    lamb = 1.0e-8
    sigma = 5120.0
    result = dict()

    solve_method = 'Direct'

    for name, method in methods.items():
        result[name] = dict()
        print(f'------------- Method: {name} -------------')
        result[name]["trace_errors"] = np.zeros((len(ks),2))
        result[name]["KRRMSE"] = np.zeros((len(ks),2))
        result[name]["KRRMAE"] = np.zeros((len(ks),2))
        result[name]["KRRSMAPE"] = np.zeros((len(ks),2))
        result[name]["queriess"] = np.zeros((len(ks),2))

        for idx_k in range(len(ks)):
            k = ks[idx_k]
            print(f'k = {k}')
            trace_err = []
            runtime = []
            queries = []
            KRRmse = []
            KRRmae = []
            KRRsmape = []
            if "Greedy" not in name:
                for i in range(num_trials):
                    while True:
                        try:
                            print(f"Trial {i}")
                            model = KRR_Nystrom(kernel = "laplace", 
                                    bandwidth = sigma)
                            model.fit_Nystrom(train_sample, train_sample_target, lamb = lamb, sample_num = k, sample_method = method, solve_method = solve_method)
                            preds = model.predict_Nystrom(test_sample)
                            break
                        except np.linalg.LinAlgError:
                            continue
                    KRRmse.append(mean_squared_error(test_sample_target, preds))
                    KRRmae.append(mean_average_error(test_sample_target, preds))
                    KRRsmape.append(SMAPE(test_sample_target, preds))
                    queries.append(model.queries)
                    trace_err.append(model.reltrace_err)  

                    print(f'KRR acc: mse {KRRmse[-1]}, mae {KRRmae[-1]}, smape {KRRsmape[-1]}')
                    print(f'time: sample {model.sample_time} s, linsolve {model.linsolve_time} s, pred {model.pred_time} s')

            else:
                model = KRR_Nystrom(kernel = "laplace", 
                            bandwidth = sigma)
                model.fit_Nystrom(train_sample, train_sample_target, lamb = lamb, sample_num = k, sample_method = method, solve_method = solve_method)
                preds = model.predict_Nystrom(test_sample)
                KRRmse.append(mean_squared_error(test_sample_target, preds))
                KRRmae.append(mean_average_error(test_sample_target, preds))
                KRRsmape.append(SMAPE(test_sample_target, preds))
                queries.append(model.queries)
                trace_err.append(model.reltrace_err) 

                print(f'KRR acc: mse {KRRmse[-1]}, mae {KRRmae[-1]}, smape {KRRsmape[-1]}')
                print(f'time: sample {model.sample_time}, linsolve {model.linsolve_time}, pred {model.pred_time}')

            result[name]["trace_errors"][idx_k,:] = [np.mean(trace_err),np.std(trace_err)]
            result[name]["KRRMSE"][idx_k,:] = [np.mean(KRRmse),np.std(KRRmse)]
            result[name]["KRRMAE"][idx_k,:] = [np.mean(KRRmae),np.std(KRRmae)]
            result[name]["KRRSMAPE"][idx_k,:] = [np.mean(KRRsmape),np.std(KRRsmape)]
            result[name]["queriess"][idx_k,:] = [np.mean(queries)/float(num_train**2),np.std(queries)/float(num_train**2)]

            savemat("data/{}_molecule100k.mat".format(name), result[name])
