"""
pw_basis.py

Author: James Ashton Nichols
Start date: June 2017

Code to deal with piece-wise linear functions on triangulations - allows for treatement of FEM solutions etc...
"""

import numpy as np
import scipy.sparse

from pyApproxTools.vector import *
from pyApproxTools.basis import *
from pyApproxTools.pw_vector import *

__all__ = ['PWBasis']

class PWBasis(Basis):
    """  A basis that knows about the PW nature of the vectors, and stores them in a flat array, for speed """

    def __init__(self, vecs=None, space='H1', is_orthonormal=False, values_flat=None, pre_allocate=0, file_name=None):
        super().__init__(vecs, space, is_orthonormal)
        
        self.values_flat = values_flat
        if vecs is not None:
            # Make a flat "values" thing for speed's sake, so we
            # can use numpy power!
            # NB we allow it to be set externally for accessing speed
            if values_flat is None:
                # Pre-allocate here is used for speed purposes... so that memory is allocated and ready to go...
                self.values_flat =  np.zeros(np.append(self.vecs[0].values.shape, max(self.n, pre_allocate)))
                for i, vec in enumerate(self.vecs):
                    self.values_flat[:,:,i] = vec.values
            else:
                if values_flat.shape[2] < self.n:
                    raise Exception('Incorrectly sized flat value matrix, are the contents correct?')
                else:
                    self.values_flat = np.zeros(np.append(self.vecs[0].values.shape, \
                                                max(self.n, pre_allocate, values_flat.shape[2]) ))
                    self.values_flat[:,:,:values_flat.shape[2]] = values_flat
                        
        elif file_name is not None:
            self.load(file_name)
    
    def add_vector(self, vec, incr_ortho=True, check_ortho=True):
        """ Add just one vector, so as to make the new Grammian calculation quick """
        super().add_vector(vec, incr_ortho=incr_ortho, check_ortho=check_ortho)

        if self.values_flat is not None:
            self.values_flat = np.pad(self.values_flat, ((0,0),(0,0),(0,self.n-self.values_flat.shape[2])), 'constant')
            self.values_flat[:,:,self.n-1] = vec.values
        else:
            self.values_flat = vec.values[:,:,np.newaxis]


    def subspace(self, indices):
        """ To be able to do "nested" spaces, the easiest way is to implement
            subspaces such that we can draw from a larger ambient space """
        return type(self)(self.vecs[indices], space=self.space, values_flat=self.values_flat[:,:,indices])

    def subspace_mask(self, mask):
        """ Here we make a subspace by using a boolean mask that needs to be of
            the same size as the number of vectors. Used for the cross validation """
        if mask.shape[0] != len(self.vecs):
            raise Exception('Subspace mask must be the same size as length of vectors')
        return type(self)(list(compress(self.vecs, mask)), space=self.space, values_flat=self.values_flat[:,:,mask])

    def reconstruct(self, c):
        # Build a function from a vector of coefficients
        u_p = type(self.vecs[0])((c * self.values_flat[:,:,:self.n]).sum(axis=2)) 
        return u_p

    def save(self, file_name):
        if self.G is not None:
            if self.S is not None and self.U is not None and self.V is not None:
                np.savez_compressed(file_name, values_flat=self.values_flat, G=self.G, S=self.S, U=self.U, V=self.V)
            else:
                np.savez_compressed(file_name, values_flat=self.values_flat, G=self.G)
        else:
            np.savez_compressed(file_name, values_flat=self.values_flat)

    def load(self, file_name):

        data = np.load(file_name)

        self.values_flat = data['values_flat']
        
        self.vecs = []
        for i in range(self.values_flat.shape[-1]):
            self.vecs.append(DyadicPWLinear(self.values_flat[:,:,i]))
        
        # TODO: make this a part of the saved file format...
        self.space = 'H1'

        if 'G' in data.files:
            self.G = data['G']
        else:
            self.G = None
        if 'S' in data.files and 'U' in data.files and 'V' in data.files:
            self.S = data['S']
            self.U = data['U']
            self.V = data['V']
        else:
            self.S = self.U = self.V = None

