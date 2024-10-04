#!/usr/bin/env python3

import numpy as np
from rpcholesky import rejection_cholesky
from time import time

def accelerated_rpqr(B, k, b = 'auto', accelerated=True, verbose=False):
    m = B.shape[0]
    n = B.shape[1]

    Q = np.zeros((m, k))
    F = np.zeros((n, k))
    rng = np.random.default_rng()

    u = np.linalg.norm(B, axis=0) ** 2

    arr_idx = np.zeros(k, dtype=int)

    if "auto" == b:
        b = int(np.ceil(k / 10))
        auto_b = True
    else:
        auto_b = False
    
    counter = 0
    while counter < k:
        idx = idx = rng.choice(range(n), size = b, p = u / sum(u), replace=True)

        if auto_b:
            start = time()
        
        C = B[:,idx] - Q[:,:counter] @ F[idx, :counter].T
        _, accepted = rejection_cholesky(C.T @ C)
        num_sel = len(accepted)
        
        if auto_b:
            rejection_time = time() - start
            start = time()

        if num_sel > k - counter:
            num_sel = k - counter
            accepted = accepted[:num_sel]

        idx = idx[accepted]
        C = C[:,accepted]
        new_counter = counter + len(accepted)
        arr_idx[counter:new_counter] = idx
        
        Q[:,counter:new_counter], _ = np.linalg.qr(C - Q[:,:counter] @ (Q[:,:counter].T @ C), mode="reduced")
        F[:,counter:new_counter] = B.T @ Q[:,counter:new_counter]

        if auto_b:
            process_time = time() - start
            target = int(np.ceil(b * process_time / (4 * rejection_time)))
            b = max([min([target, int(np.ceil(1.5*b)), int(np.ceil(k/3))]),
                     int(np.ceil(b/3)), 10])


        u -= np.linalg.norm(F[:,counter:new_counter], axis=1)**2
        u = u.clip(min = 0)

        counter = new_counter

        if verbose:
            print("Accepted {} / {}".format(num_sel, b))

    return Q, F, arr_idx

def rpqr(B, k, **kwargs):
    return accelerated_rpqr(B, k, **kwargs)
