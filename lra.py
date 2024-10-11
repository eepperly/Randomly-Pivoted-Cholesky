#!/usr/bin/env python3

import numpy as np
import scipy as sp

from abc import ABC, abstractmethod

class AbstractPSDLowRank(ABC):

    def __init__(self, **kwargs):
        self.idx = kwargs['idx'] if ('idx' in kwargs) else None
        self.rows = kwargs['rows'] if ('rows' in kwargs) else None

    @abstractmethod
    def trace(self):
        pass

    @abstractmethod
    def __matmul__(self, other):
        pass

    def __rmatmul__(self, other):
        return (self @ other.T).T

    @abstractmethod
    def rank(self):
        pass

    @abstractmethod
    def eigenvalue_decomposition(self):
        pass

    def matrix(self):
        return self @ np.identity(self.shape[0])

    def get_rows(self):
        if not (self.rows is None):
            return self.rows
        raise RuntimeError("Rows are not defined for this low-rank approximation")

    def get_indices(self):
        if not (self.idx is None):
            return self.idx
        raise RuntimeError("Indices are not defined for this low-rank approximation")
    
class CompactEigenvalueDecomposition(AbstractPSDLowRank):

    def __init__(self, V, Lambda, **kwargs):
        super().__init__(**kwargs)
        self.V = V
        self.Lambda = Lambda
        self.shape = (self.V[0], self.V[0])

    @staticmethod
    def from_G(G, **kwargs):
        Q, R = np.linalg.qr(G.T, "reduced")
        U, S, _ = np.linalg.svd(R)
        return CompactEigenvalueDecomposition(Q @ U, S ** 2, **kwargs)

    def evals(self):
        return self.Lambda

    def evecs(self):
        return self.V

    def trace(self):
        return sum(self.Lambda)

    def __matmul__(self, other):
        return self.V @ (self.Lambda * (self.V.T @ other))

    def rank(self):
        return len(S)

    def matrix(self):
        return self.V @ (self.Lambda * self.V.T)

    def eigenvalue_decomposition(self):
        return self

    def krr(self, b, lam):
        VTb = self.V.T @ b
        return b / lam + self.V @ (np.divide(VTb, self.Lambda.reshape(VTb.shape) + lam) - VTb / lam)
    
    def krr_vec(self,b,lam):
        VTb = self.V.T @ b
        diag = self.Lambda.reshape(VTb.shape[0]) + lam
        return b / lam + self.V @ (np.divide(VTb, diag[:,np.newaxis]) - VTb / lam)
        
class PSDLowRank(AbstractPSDLowRank):

    def __init__(self, G, **kwargs):
        super().__init__(**kwargs)
        self.G = G
        self.shape = (G.shape[1],G.shape[1])

    def trace(self):
        return np.linalg.norm(self.G, 'fro')**2

    def __matmul__(self,other):
        return self.G.T @ (self.G @ other)

    def rank(self):
        return self.G.shape[0]

    def matrix(self):
        return self.G.T @ self.G

    def eigenvalue_decomposition(self):
        return CompactEigenvalueDecomposition.from_G(self.G, idx = self.idx, rows = self.rows)

    def scale(self, scaling):
        return PSDLowRank(self.G * scaling[np.newaxis,:], idx = self.idx, rows = self.rows)

    def get_left_factor(self):
        return self.G.T

    def get_right_factor(self):
        return self.G

class NystromExtension(AbstractPSDLowRank):

    def __init__(self, core, factor = None, **kwargs):
        super().__init__(**kwargs)
        if "rows" not in kwargs:
            raise RuntimeError("Need to specify rows for Nystrom extension")
        self.C = (core+core.T)/2
        self.shape = (self.rows.shape[1], self.rows.shape[1])
        self.G = factor

    def get_factor(self):
        if self.G is None:
            L = np.linalg.cholesky(self.C+np.trace(self.C)*np.finfo(float).eps*np.identity(self.C.shape[0]))
            self.G = np.linalg.solve(L, self.rows)
        return self.G

    def get_left_factor(self):
        return self.get_factor().T

    def get_right_factor(self):
        return self.get_factor()
        
    def trace(self):
        return np.linalg.norm(self.get_factor())**2

    def __matmul__(self, other):
        return self.rows.T @ np.linalg.solve(self.C, self.rows @ other)

    def rank(self):
        return self.C.shape[0]

    def matrix(self):
        return self.rows.T @ np.linalg.solve(self.C, self.rows)
    
    def eigenvalue_decomposition(self):
        return CompactEigenvalueDecomposition.from_G(self.get_factor(), idx = self.idx, rows = self.rows)

    def scale(self, scaling):
        return NystromExtension(self.C, idx = self.idx, rows = self.rows * scaling[np.newaxis,:])
