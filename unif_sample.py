#!/usr/bin/env python3

import numpy as np
import random
from lra import NystromExtension 
from scipy.linalg import solve_triangular

def uniform_sample(A, k):
    n = A.shape[0]
    sample = np.random.choice(range(n), k, False)
    return lra_from_sample(A, sample)
