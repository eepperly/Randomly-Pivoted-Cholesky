#!/usr/bin/env python3

import numpy as np
from abc import ABC, abstractmethod
import numbers
from kernels import MaternKernel, MaternKernel_vec, MaternKernel_mtx, GaussianKernel, GaussianKernel_vec, GaussianKernel_mtx, LaplaceKernel, LaplaceKernel_vec, LaplaceKernel_mtx
from functools import partial

class AbstractPSDMatrix(ABC):

    def __init__(self, **kwargs):
        self.queries = 0
        self.count_queries = kwargs['count_queries'] if ('count_queries' in kwargs) else True
        self.dpp_stuff = None

    @abstractmethod
    def _diag_helper(self, *args):
        pass

    @abstractmethod 
    def _getitem_helper(self, *args):
        pass 
    
    def __getitem__(self, *args):
        to_return = self._getitem_helper(*args)
        if isinstance(to_return, np.ndarray):
            self.queries += to_return.size
        else:
            self.queries += 1
        return to_return

    def diag(self, *args):
        to_return = self._diag_helper(*args)
        if isinstance(to_return, np.ndarray):
            self.queries += to_return.size
        else:
            self.queries += 1
        return to_return
    
    def __call__(self, *args):
        return self.__getitem__(*args)

    def trace(self):
        return sum(self.diag())

    def num_queries(self):
        return self.queries

    def reset(self):
        self.queries = 0
        self.dpp_stuff = None

    def to_matrix(self):
        return PSDMatrix(self[:,:])
        
class PSDMatrix(AbstractPSDMatrix):

    def __init__(self, A, **kwargs):
        super().__init__(**kwargs)
        self.matrix = A
        self.shape = A.shape

    def _diag_helper(self, *args):
        if len(args) > 0:
            X = list(args[0])
        else:
            X = range(self.matrix.shape[0])
        return self.matrix[X,X]

    def _getitem_helper(self, *args):
        return self.matrix.__getitem__(*args)

class FunctionMatrix(AbstractPSDMatrix):

    def __init__(self, n, **kwargs):
        self.shape = (n,n)
        super().__init__(**kwargs)
    
    @abstractmethod
    def _function(self,i,j):
        pass
    
    @abstractmethod
    def _function_vec(self,vec_i,vec_j):
        pass

    @abstractmethod
    def _function_mtx(self,vec_i,vec_j):
        pass
    
    def _diag_helper(self, *args):
        if len(args) > 0:
            idx = list(args[0])
        else:
            idx = range(self.shape[0])
        return self._function_vec(idx,idx)
    
    def _getitem_helper(self, *args):
        idx = args[0]
        if len(idx) == 1:
            idx = [idx,idx]
        else:
            idx = list(idx)

        for j in range(2):
            if isinstance(idx[j], np.ndarray):
                idx[j] = idx[j].ravel().tolist()
            elif isinstance(idx[j], slice):
                idx[j] = list(range(self.shape[0]))[idx[j]]
            elif isinstance(idx[j], list):
                pass
            elif isinstance(idx[j], numbers.Integral):
                idx[j] = [idx[j]]
            else:
                raise RuntimeError("Indexing not implemented with index of type {}".format(type(idx)))

        if len(idx[0]) == 1 and len(idx[1]) == 1:
            return self._function(idx[0], idx[1])
        else:
            mtx = self._function_mtx(idx[0],idx[1])
            if len(idx[0]) == 1: return mtx[0,:]
            elif len(idx[1]) == 1: return mtx[:,0]
            else: return mtx            
    
class KernelMatrix(FunctionMatrix):
    @staticmethod
    def kernel_from_input(kernel, bandwidth = 1.0, **kwargs):
        if isinstance(kernel, str):
            if kernel == 'gaussian':
                return partial(GaussianKernel,bandwidth=bandwidth), partial(GaussianKernel_vec, bandwidth=bandwidth), partial(GaussianKernel_mtx, bandwidth=bandwidth)
            elif kernel == 'matern':
                return partial(MaternKernel,bandwidth=bandwidth,nu=kwargs["nu"]), partial(MaternKernel_vec,bandwidth=bandwidth,nu=kwargs["nu"]), partial(MaternKernel_mtx,bandwidth=bandwidth,nu=kwargs["nu"])
            elif kernel == 'laplace':
                return partial(LaplaceKernel,bandwidth=bandwidth), partial(LaplaceKernel_vec, bandwidth=bandwidth), partial(LaplaceKernel_mtx, bandwidth=bandwidth)
                
            else:
                raise RuntimeError("Kernel name {} not recognized".format(kernel))
        else:
            return kernel
    
    def __init__(self, X, kernel = "gaussian", bandwidth = 1.0, **kwargs):
        super().__init__(X.shape[0],**kwargs)
        self.data = X
        kernel, kernel_vec, kernel_mtx = KernelMatrix.kernel_from_input(kernel, bandwidth = bandwidth, **kwargs)
        self.kernel = kernel
        self.kernel_vec = kernel_vec
        self.kernel_mtx = kernel_mtx
        
        
    def _function(self, i, j):
        return self.kernel(self.data[i,:], self.data[j,:])
    
    def _function_vec(self,vec_i,vec_j):
        return self.kernel_vec(self.data[vec_i,:], self.data[vec_j,:])
    
    def _function_mtx(self,vec_i,vec_j):
        return self.kernel_mtx(self.data[vec_i,:], self.data[vec_j,:])

class NonsymmetricKernelMatrix(object):
    
    def __init__(self, X, Y, kernel = "gaussian", bandwidth = 1.0, **kwargs):
        self.X = X
        self.Y = Y
        self.shape = (X.shape[0], Y.shape[0])
        self.kernel, self.kernel_vec, self.kernel_mtx = KernelMatrix.kernel_from_input(kernel, bandwidth = bandwidth, **kwargs)
        
    def _function(self, i, j):
        return self.kernel(self.X[i,:], self.Y[j,:])
    
    def _function_vec(self,vec_i,vec_j):
        return self.kernel_vec(self.X[vec_i,:], self.Y[vec_j,:])
    
    def _function_mtx(self,vec_i,vec_j):
        return self.kernel_mtx(self.X[vec_i,:], self.Y[vec_j,:])

    def _getitem_helper(self, *args):
        idx = args[0]
        if len(idx) == 1:
            idx = [idx,idx]
        else:
            idx = list(idx)

        for j in range(2):
            if isinstance(idx[j], np.ndarray):
                idx[j] = idx[j].ravel().tolist()
            elif isinstance(idx[j], slice):
                idx[j] = list(range(self.shape[j]))[idx[j]]
            elif isinstance(idx[j], list):
                pass
            elif isinstance(idx[j], numbers.Integral):
                idx[j] = [idx[j]]
            else:
                raise RuntimeError("Indexing not implemented with index of type {}".format(type(idx)))

        if len(idx[0]) == 1 and len(idx[1]) == 1:
            return self._function(idx[0], idx[1])
        else:
            mtx = self._function_mtx(idx[0],idx[1])
            if len(idx[0]) == 1: return mtx[0,:]
            elif len(idx[1]) == 1: return mtx[:,0]
            else: return mtx
            
    def __getitem__(self, *args):
        return self._getitem_helper(*args)
