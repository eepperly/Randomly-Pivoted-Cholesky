#!/usr/bin/env python3

import numpy as np
from matrix import PSDMatrix, KernelMatrix
import itertools
import scipy.io
import re
from warnings import warn
import os

def smile(N, bandwidth = 2.0, **kwargs):
    small = int(np.ceil(N ** (1.0/2)))
    eye_points = kwargs["eye_points"] if ("eye_points" in kwargs) else small
    mouth_points = kwargs["mouth_points"] if ("mouth_points" in kwargs) else int(np.ceil(N/10.0))
    face_points = N - 2 * eye_points - mouth_points

    X = np.zeros((N, 2))
    idx = 0

    # Eyes
    for x_shift in [-4.0, 4.0]:
        for i in range(eye_points):
            while True:
                x = 2 * np.random.rand() - 1
                y = 2 * np.random.rand() - 1
                if x**2 + y**2 <= 1.0:
                    X[idx, 0] = x + x_shift
                    X[idx, 1] = y + 4.0
                    idx += 1
                    break

    # Mouth
    for x in list(np.linspace(-5.0, 5.0, mouth_points)):
        X[idx, 0] = x
        X[idx, 1] = x**2 / 16.0 - 5.0
        idx += 1

    # Face
    for theta in list(np.linspace(0, 2*np.pi, face_points)):
        X[idx, 0] = 10.0 * np.cos(theta)
        X[idx, 1] = 10.0 * np.sin(theta)
        idx += 1

    if "dimension" in kwargs:
        d = kwargs["dimension"]
        Q, _ = np.linalg.qr(np.random.randn(d, 2))
        Q = Q.T
        X = X @ Q

    return KernelMatrix(X, bandwidth = bandwidth, **kwargs)    

def expspiral(N, rate=1e-3, rotate_rate=5e-3, power = 1.5, bandwidth = 0.02):
    X = np.zeros((N, 2))
    t = np.array(range(N))
    X[:,0] = np.exp(-rate * N * (t/N) ** power) * np.cos(rotate_rate * t)
    X[:,1] = np.exp(-rate * N * (t/N) ** power) * np.sin(rotate_rate * t)
    return KernelMatrix(X, bandwidth = bandwidth)

def robspiral(N):
    times = np.linspace(0, 2, N)
    times = times ** 6
    times = times[::-1]
    x = np.exp(.2 * times) * np.cos(times)
    y = np.exp(.2 * times) * np.sin(times)
    X = np.column_stack((x,y))
    bandwidth = 1000
    return KernelMatrix(X, bandwidth = bandwidth)

def powerspiral(N, max_radius = 1.0, angle = 0.2, decay_power = 1, bandwidth = 1e-3):
    X = np.zeros((N, 2))
    t = np.array(range(1,N+1), dtype=float)
    radii = max_radius * t ** (-decay_power)
    angles = angle * t
    X[:,0] = radii * np.cos(angles)
    X[:,1] = radii * np.sin(angles)
    # np.random.shuffle(X)
    return KernelMatrix(X, bandwidth = bandwidth)

def outliers(N, num_outliers = 50):
    X = 0.5*np.random.randn(N, 20)/np.sqrt(20.0)
    X[np.random.choice(range(N), size = num_outliers, replace = False),:] += 100.0 * np.random.randn(num_outliers, 20)
    return KernelMatrix(X)

def random_kernel_matrix(N, d = 10, **kwargs):
    X = np.random.randn(N, d)
    return KernelMatrix(X, **kwargs)

def kernel_from_data(data, **kwargs):
    mean = np.mean(data, axis=0)
    stddev = 0.0
    for i in range(data.shape[0]):
        stddev += np.linalg.norm(data[i,:] - mean)**2
    return KernelMatrix(data, bandwidth = np.sqrt(stddev / data.shape[0]), **kwargs)

def find_mat_files_and_prefixes(folder_path):
    mat_files = []
    mat_prefixes = []
    pattern = re.compile(r'(.*)\.mat$')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            match = pattern.match(file)
            if match:
                mat_files.append(os.path.join(root, file))
                mat_prefixes.append(match.group(1))
    return mat_files, mat_prefixes

def gallery(N, datafolder = None, min_N=None):
    if min_N is None:
        min_N = 0
    
    yield "Smile (high)", smile(N, bandwidth=0.2, extra_stability=True)
    yield "Smile (medium)", smile(N, bandwidth=0.15, extra_stability=True)
    yield "Smile (low)", smile(N, bandwidth=0.1, extra_stability=True)
    yield "Exponential Spiral (high)", expspiral(N,bandwidth=0.02)
    yield "Exponential Spiral (medium)", expspiral(N,bandwidth=0.015)
    yield "Exponential Spiral (low)", expspiral(N,bandwidth=0.01)
    yield "Outliers (50)", outliers(N)
    yield "Outliers (500)", outliers(N, num_outliers=500)
    yield "Outliers (5000)", outliers(N, num_outliers=5000)
    
    for d in [2,10,100,1000]:
        for bandwidth, bandname in [(2*np.sqrt(d),"high"),(np.sqrt(d),"medium"),(np.sqrt(d)/2,"low")]:
            for kerneltype in ["laplace", "matern", "gaussian"]:
                yield f"Random ({d},{bandname},{kerneltype})", random_kernel_matrix(N, d=d, bandwidth=(bandwidth/10 if kerneltype=="gaussian" else bandwidth), kernel=kerneltype, nu=1.5)

    if datafolder is None:
        return
    
    files, names = find_mat_files_and_prefixes(datafolder)
    for name, myfile in zip(names, files):
        data = scipy.io.loadmat(myfile)
        if "Xtr" in data:
            X = data["Xtr"]
        elif "A" in data:
            X = data["A"]
        else:
            print(f"Dataset {name} did not contain a variable named 'Xtr' or 'A'. Skipping")
            continue
        if type(X) != np.ndarray:
            X = X.toarray()
        X = X.astype(float)

        if X.shape[0] < min_N:
            print(f"Dataset {name} of size {X.shape[0]} is too small. Skipping")
            continue

        if X.shape[0] > N:
            X = X[np.random.choice(X.shape[0], N, replace=False),:]
        
        for kerneltype in ["laplace", "matern", "gaussian"]:
            A = KernelMatrix(X, kerneltype, bandwidth = "approx_median", nu=1.5)
            yield f"{name} ({kerneltype})", A
