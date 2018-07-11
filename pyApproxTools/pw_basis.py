"""
pw_basis.py

Author: James Ashton Nichols
Start date: June 2017

Code to deal with piece-wise linear functions on triangulations - allows for treatement of FEM solutions etc...
"""

import numpy as np
import scipy.sparse
import itertools
import random

from pyApproxTools.vector import *
from pyApproxTools.basis import *
from pyApproxTools.pw_vector import *

__all__ = ['PWBasis']

class PWBasis(Basis):
    """  A basis that knows about the PW nature of the vectors, and stores them in a flat array, for numpy enhanced speed """

    def __init__(self, vecs=None, G=None, values_flat=None, pre_allocate=0, file_name=None):
        super().__init__(vecs=vecs, G=G)
        
        self._values_flat = values_flat
        if vecs is not None:
            # Make a flat "values" thing for speed's sake, so we
            # can use numpy power!
            # NB we allow it to be set externally for accessing speed
            if values_flat is None:
                # Pre-allocate here is used for speed purposes... so that memory is allocated and ready to go...
                self._values_flat =  np.zeros(np.append(self._vecs[0].values.shape, max(self.n, pre_allocate)))
                for i, vec in enumerate(self._vecs):
                    self._values_flat[:,:,i] = vec.values
            else:
                if values_flat.shape[2] < self.n:
                    raise Exception('Incorrectly sized flat value matrix, are the contents correct?')
                else:
                    self._values_flat = np.zeros(np.append(self._vecs[0].values.shape, \
                                                max(self.n, pre_allocate, values_flat.shape[2]) ))
                    self._values_flat[:,:,:values_flat.shape[2]] = values_flat
        
        elif file_name is not None:
            self.load(file_name)
   
    def __getitem__(self, idx):
        if self._is_herm_trans:
            raise IndexError(self.__name__ + ': can not access rows of Hermitian of basis')

        if isinstance(idx, int):
            return self._vecs[idx]

        elif isinstance(idx, slice):
            sub = type(self)(self._vecs[idx], values_flat=self._values_flat[:,:,idx])
            if not np.any(self._vec_is_new):
                sub._G = self._G[idx, idx]
            return sub

        elif hasattr(idx, '__len__') and len(idx) == len(self): 
            # NOT CONFIDENT THAT THIS WORKS... but in this case idx is a boolean mask...
            sub = type(self)(list(itertools.compress(self._vecs, idx)), values_flat=self._values_flat[:,:,idx])
            if not np.any(self._vec_is_new):
                sub._G = self._G[idx,idx]
            return sub

        else:
            raise TypeError(self.__class__.__name__ + ': idx type incorrect')

    def __setitem__(self, i, v):
        super().__setitem__(i, v)
        if isinstance(v, type(self._vecs[0])):
            self._values_flat[:,:,i] = v.values
        elif isinstance(v, type(PWBasis)):
            self._values_flat[:,:,i] = v._values_flat[:,:,i]

    def add_vec(self, vec):
        """ Add just one vector, so as to make the new Grammian calculation quick """
        super().add_vec(vec)

        if self._values_flat is not None:
            self._values_flat = np.pad(self._values_flat, ((0,0),(0,0),(0,self.n-self._values_flat.shape[2])), 'constant')
            self._values_flat[:,:,self.n-1] = self._vecs[-1].values
        else:
            self._values_flat = self._vecs[-1].values[:,:,np.newaxis]

    def append(self, other):
        """ Add just one vector, so as to make the new Grammian calculation quick """
        super().append(other)
    
        if self._values_flat is not None:
            old_n = self._values_flat.shape[2]
            self._values_flat = np.pad(self._values_flat, ((0,0),(0,0),(0,self.n-self._values_flat.shape[2])), 'constant')
        else:
            old_n = 0
            self._values_flat = np.zeros(np.append(self._vecs[0].values.shape, len(other)))
    
        for i in range(old_n, self.n):
            self._values_flat[:,:,i] = self._vecs[i].values

    def _matmul(self, other):

        if len(other) != self.n:
            raise ValueError('{0}: array-like length {1} not equal to length of basis {2}'.format(self.__class__.__name__, len(other), self.n))

        # Two cases: right is a list, a vector, or is a matrix. 
        # For now we don't support tensors (but that would be good...)
        if isinstance(other, (np.ndarray, np.generic)):
            if other.ndim == 1:
                return self._colmul(other)
            elif other.ndim == 2:
                new = self._values_flat[:,:,:self.n] @ other
                vecs = []
                for i in range(other.shape[1]):
                    vecs.append(type(self._vecs[0])(new[:,:,i]))

                return type(self)(vecs, values_flat = new)
            else:
                raise ValueError('{0}: matmul only supports vectors and matrices'.format(self.__class__.__name__))

        elif isinstance(other, list):
            return self._colmul(other)
        else:
            raise ValueError('{0}: unsupported @ compatibility'.format(self.__class__.__name__))
        
    def _colmul(self, c):
        """ Build a function from a vector of coefficients """
        u_p = type(self._vecs[0])((c * self._values_flat[:,:,:self.n]).sum(axis=2)) 
        return u_p

    def shuffle_vectors(self):

        random.shuffle(self._vecs)
        for i, vec in enumerate(self._vecs):
            self._values_flat[:,:,i] = vec.values
        self._G = None

    def save(self, file_name):
        if not np.any(self._vec_is_new):
            if self._S is not None and self_.U is not None and self._V is not None:
                np.savez_compressed(file_name, values_flat=self._values_flat, G=self._G, S=self._S, U=self._U, V=self._V)
            else:
                np.savez_compressed(file_name, values_flat=self._values_flat, G=self._G)
        else:
            np.savez_compressed(file_name, values_flat=self._values_flat)

    def load(self, file_name):

        data = np.load(file_name)

        self._values_flat = data['values_flat']
        
        self._vecs = []
        for i in range(self._values_flat.shape[-1]):
            self._vecs.append(PWLinearSqDyadicH1(self._values_flat[:,:,i]))
        
        if 'G' in data.files:
            self._G = data['G']
        else:
            self._G = None
        if 'S' in data.files and 'U' in data.files and 'V' in data.files:
            self._S = data['S']
            self._U = data['U']
            self._V = data['V']
        else:
            self._S = self._U = self._V = None

