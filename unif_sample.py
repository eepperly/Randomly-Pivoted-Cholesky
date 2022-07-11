#!/usr/bin/env python3

import numpy as np
from utils import lra_from_sample

def uniform_sample(A, k):
    n = A.shape[0]
    sample = np.random.choice(range(n), k, False)
    return lra_from_sample(A, sample)
