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

import qml, os
from scipy.io import savemat, loadmat
import numpy as np

def get_molecules(directory = "molecules/", max_atoms = 29, max_mols = np.Inf, output_index = 7):
    compounds = []
    energies = []
    for f in sorted(os.listdir("molecules/")):
        if len(compounds) >= max_mols:
            break

        try:
            mol = qml.Compound(xyz="molecules/"+f)
            mol.generate_coulomb_matrix(size=max_atoms, sorting="row-norm")
            with open("molecules/"+f) as myfile:
                line = list(myfile.readlines())[1]
                energies.append(float(line.split()[output_index]) * 27.2114) # Hartrees to eV
            compounds.append(mol)
        except ValueError:
            pass
    
    c = list(zip(compounds, energies))
    np.random.shuffle(c)
    compounds, energies = zip(*c)

    X = np.array([mol.representation for mol in compounds])
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
    import rp_cholesky
    import leverage_score
    import unif_sample
    import matplotlib.pyplot as plt
    import time
    from functools import partial

    methods = { 'Greedy' : rp_cholesky.greedy,
                'Uniform' : unif_sample.uniform_sample,
                'RPCholesky' : rp_cholesky.rp_cholesky,
                'RLS' : leverage_score.recursive_rls_acc,
                'block50RPCholesky' : partial(rp_cholesky.block_rp_cholesky,b=50) }

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
                            model = KRR_Nystrom(kernel = "gaussian", 
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
