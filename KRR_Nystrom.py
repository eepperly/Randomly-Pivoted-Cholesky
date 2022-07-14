#!/usr/bin/env python

import numpy as np
from matrix import KernelMatrix, NonsymmetricKernelMatrix
import scipy
import time
from scipy.linalg import solve_triangular

class KRR_Nystrom():
    def __init__(self,
                 kernel = "gaussian", 
                 bandwidth = 1):
        self.kernel = kernel
        self.bandwidth = bandwidth
    
    def fit(self, Xtr, Ytr, lamb):
        self.Xtr = Xtr
        ts = time.time()
        self.K = KernelMatrix(Xtr, kernel = self.kernel, bandwidth = self.bandwidth)
        K_exact = self.K[:,:]
        self.sol = scipy.linalg.solve(K_exact+lamb*np.shape(K_exact)[0]*np.eye(np.shape(K_exact)[0]),Ytr, assume_a='pos')
        te = time.time()
        self.linsolve_time = te - ts
        
    def predict(self, Xts):
        ts = time.time()
        K_pred = NonsymmetricKernelMatrix(Xts, self.Xtr, kernel = self.kernel, bandwidth = self.bandwidth)
        K_pred_exact = K_pred[:,:]
        preds = K_pred_exact @ self.sol
        te = time.time()
        self.pred_time = te - ts
        return preds
    
    def fit_Nystrom(self, Xtr, Ytr, lamb, sample_num, sample_method, solve_method = 'Direct', ite = 400):
        self.Xtr = Xtr
        self.K = KernelMatrix(Xtr, kernel = self.kernel, bandwidth = self.bandwidth)
        ts = time.time()
        lra = sample_method(self.K, sample_num)
        arr_idx = lra.idx
        KMn = lra.rows
        te = time.time()
        self.sample_idx = arr_idx
        self.sample_time = te - ts
        self.queries = self.K.num_queries()
        
        trK = self.K.trace()
        self.reltrace_err = (trK-lra.trace())/ trK
        
        self.K.reset_queries()
        ts = time.time()
        if solve_method == 'Direct':
            KMM = KMn[:,arr_idx]
            KnM = KMn.T
            self.sol = scipy.linalg.solve(KMn @ KnM + KnM.shape[0]*lamb*KMM + 100*KMM.max()*np.finfo(float).eps*np.identity(sample_num), KMn @ Ytr, assume_a='pos')
            
ite) for i in range(np.shape(Ytr)[1])])
        te = time.time()
        self.linsolve_time = te - ts
        
    def predict_Nystrom(self, Xts):
        ts = time.time()
        K_pred = NonsymmetricKernelMatrix(Xts, self.Xtr, kernel = self.kernel, bandwidth = self.bandwidth)
        KtM =  K_pred[:,self.sample_idx]
        preds = KtM @ self.sol
        te = time.time()
        self.pred_time = te - ts
        return preds
