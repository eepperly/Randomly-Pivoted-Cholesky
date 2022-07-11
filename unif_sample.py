#!/usr/bin/env python3

import numpy as np
import random
from lra import NystromExtension 
from scipy.linalg import solve_triangular

def uniform_sample(A, n, k):
    sample = np.random.choice(range(n), k, False)
    F = A[:,sample]
    C = F[sample,:]
    Lc = np.linalg.cholesky(C+100*C.max()*np.finfo(float).eps*np.identity(k))
    factor = solve_triangular(Lc, F.T,lower=True).T 
    
    return NystromExtension(F, C, factor=factor, idx = sample, rows = F.T)

