"""
basis.py

Author: James Ashton Nichols
Start date: June 2017

The abstraction of some linear algebra in Hilbert spaces for doing functional analysis 
computations, where the class "Basis" does some predictable operations on the 
class "Vector", which has to have a dot product defined as well as typical linear algebra.
"""

import math
import numpy as np
import scipy as sp
import scipy.sparse 
import scipy.linalg
import random
import copy
import warnings
import itertools

from pyApproxTools.vector import *

__all__ = ['Basis', 'BasisPair', 'FavorableBasisPair']

_ORTHO_MATRIX_TOL = 1e-6
_ORTHO_VECTOR_TOL = 1e-10

class Basis(object):
    """ Class representing the mathematical concept of a Basis. Routines available
    include:

    subspace(indices)
    subspace_mask(mask)
    make_grammian()
    cross_grammian(other)
    project(u)
    matrix_multiply(A)
    orthonormalise()
    """

    def __init__(self, vecs=None, G=None):
        
        self._vecs = vecs or []
        self._is_herm_trans = False
    
        # CAUTION: We don't at any time check if G is correct. So... make sure it's set correctly
        self._G = G
        if G is None:
            self._vec_is_new = np.full(self.n, True)
        else:
            self._vec_is_new = np.full(self.n, False)

        self._is_orthonormal = None

        self._L_inv = None
        self._U = self._S = self._V = None
    
    # TODO: Lets create explicit orthonormal factory constructors... including for the various basis types that
    # are implemented in the utils file
    #@classmethod
    #def orthonormal(cls, vecs):

    @property
    def n(self):
        return len(self._vecs)
    def __len__(self):
        return len(self._vecs)

    @property
    def G(self):
        if np.all(self._vec_is_new):
            self._G = np.zeros([self.n,self.n])
            for i in range(self.n):
                for j in range(i, self.n):
                    self._G[i,j] = self._G[j,i] = self._vecs[i].dot(self._vecs[j])
            self._vec_is_new = np.full(self.n, False)

        elif np.any(self._vec_is_new):       
            for i in range(self.n):
                if self._vec_is_new[i]:
                    for j in range(self.n):
                        self._G[i,j] = self._G[j,i] = self._vecs[i].dot(self._vecs[j]) 
            self._vec_is_new = np.full(self.n, False)

        return self._G
    
    @property
    def is_orthonormal(self):
        if np.all(~self._vec_is_new): # MAYBE NOT DO THIS AND CALL G ANYHOW
            if self._is_orthonormal is not None:
                return self._is_orthonormal
            
        if (self.G - np.eye(self.n)).max() / self.n <= _ORTHO_MATRIX_TOL:
            self._is_orthonormal = True
            return True
        else:
            self._is_orthonormal = False
            return False

    @property
    def H(self):
        herm = copy.copy(self)
        herm._is_herm_trans = not self._is_herm_trans
        return herm

    def __getitem__(self, idx):
        if self._is_herm_trans:
            raise IndexError(self.__class__.__name__ + ': can not access rows of Hermitian of basis')

        sel = None
        if isinstance(idx, int):
            sel = self._vecs[idx]
        elif isinstance(idx, slice):
            sel = self._vecs[indices]
        elif hasattr(idx, '__len__') and len(idx) == len(self): # NOT CONFIDENT THAT THIS WORKS...
            sel = list(itertools.compress(self._vecs, idx))
        
        if sel is not None:
            if len(sel) > 1:
                sub = type(self)(sel)
                
                if not np.any(self._vec_is_new):
                    sub._G = self._G[idx, idx]
            else: # The case where therer is only one vector selected... we return the vector, not a basis (should we though ??)
                sub = sel[0]
            return sub
        else:
            return TypeError(self.__class__.__name__ + ': idx type incorrect')

    def __setitem__(self, i, v):
        if self._is_herm_trans:
            raise IndexError(self.__class__.__name__ + ': can not access rows of Hermitian of basis')

        if isinstance(v, type(self._vecs[i])): 
            self._vecs[i] = v
            self._vec_is_new[i] = True
        elif isinstance(v, (list, tuple)) or isinstance(v, Basis):
            if isinstance(v[0], type(self._vecs[0])):
                self._vecs[i] = v
                self._vec_is_new[i] = True
            else:
                raise TypeError(self.__class__.__name__ + ': vectors of list or basis of incorrect type')
        else: 
            raise TypeError(self.__class__.__name__ + ': type mismatch of vector to insert')
            
    def __iter__(self):
        return iter(self._vecs)

    def __matmul__(self, other):        
        if self._is_herm_trans:
            return self._trans_matmul(other)
        else:
            return self._matmul(other)

    def matmul(self, other):
        return self.__matmul__(other)

    def _matmul(self, other):

        if len(other) != self.n:
            raise ValueError('{0}: array-like length {1} not equal to length of basis {2}'.format(self.__class__.__name__, len(other), self.n))

        # Two cases: right is a list, a vector, or is a matrix. 
        # For now we don't support tensors (but that would be good...)
        if isinstance(other, (np.ndarray, np.generic)):
            if other.ndim == 1:
                return self._colmul(other)
            elif other.ndim == 2:
                vecs = []
                for i in range(other.shape[1]):
                    vecs.append(self._colmul(other[:,i]))
                return type(self)(vecs)
            else:
                raise ValueError('{0}: matmul only supports vectors and matrices'.format(self.__class__.__name__))

        elif isinstance(other, (list, tuple)):
            return self._colmul(other)
        else:
            raise ValueError('{0}: unsupported @ compatibility'.format(self.__class__.__name__))
        
    def _colmul(self, c):
        u_p = type(self._vecs[0])()
        for i, c_i in enumerate(c):
            if c_i != 0:
                u_p += c_i * self._vecs[i] 
        return u_p

    def _trans_matmul(self, other):
        if isinstance(other, type(self._vecs[0])):
            return self.dot(other)
        
        elif isinstance(other, Basis):
            if not other._is_herm_trans:
                return self.cross_grammian(other)
            else:
                raise ValueError(self.__class__.__name__ + ': left basis must be transposed and right not')

        else:
            raise ValueError('{0}: unsupported @ compatibility'.format(self.__class__.__name__))

    def __rmatmul__(self, other):
        if self._is_herm_trans:
            return self._matmul(other)
        else:
            return self._trans_rmatmul(other)

    def rmatmul(self, other):
        return self.__rmatmul__(other)

    def _trans_rmatmul(self, other):
        if isinstance(other, type(self._vecs[0])):
            return self.dot(other)
        
        elif isinstance(other, Basis):
            if other._is_herm_trans:
                return self.cross_grammian(other).T
            else:
                raise ValueError(self.__class__.__name__ + ': left basis must be transposed and right not')

    def dot(self, u):
        u_d = np.zeros(self.n)
        for i, v in enumerate(self):
            u_d[i] = v.dot(u)
        return u_d

    def cross_grammian(self, other):
        CG = np.zeros((self.n, other.n))
        for i, v in enumerate(self):
            for j, w in enumerate(other):
                CG[i,j] = v.dot(w)

        return CG

    def add_vec(self, vec):
        """ Add just one vector, so as to make the new Grammian calculation quick """
          
        vec = copy.deepcopy(vec)
        
        self._vecs.append(vec)
        self._G = np.pad(self.G, ((0,1),(0,1)), 'constant')
        self._vec_is_new = np.pad(self._vec_is_new, (0,1), 'constant', constant_values=True)

        if np.any(self._vec_is_new):
            self._vec_is_new = np.pad(self._vec_is_new, (0,1), 'constant', constant_values=True)
        else:
            self._vec_is_new = np.pad(self._vec_is_new, (0,1), 'constant', constant_values=False)

            for i in range(self.n):
                self._G[self.n-1, i] = self._G[i, self.n-1] = self._vecs[-1].dot(self._vecs[i])

        self._U = self._V = self._S = None
   
    def append(self, other):
        """ add multiple vectors, no Grammian update """

        if isinstance(other, type(self[0])):
            self.add_vec(other)
        elif isinstance(other, Basis) and isinstance(other[0], type(self[0])):
            self._vecs.extend(other._vecs)
        
            self._G = np.pad(self.G, ((0,len(other)),(0,len(other))), 'constant')
            self._vec_is_new = np.pad(self._vec_is_new, (0,len(other)), 'constant', constant_values=True)

            self._U = self._V = self._S = None

        elif isinstance(other, list) and isinstance(other[0], type(self[0])):
            self._vecs.extend(other)

            self._G = np.pad(self.G, ((0,len(other)),(0,len(other))), 'constant')
            self._vec_is_new = np.pad(self._vec_is_new, (0,len(other)), 'constant', constant_values=True)

            self._U = self._V = self._S = None

    def add_vec_orthogonalise(self, vec):
        """ add a vector - if it is orthonormal already just add it, other wise do one gram-schmidt step,
            which if the basis is already orthogonal, will result in an augmented orthonormal basis """
        vec = vec - self.project(vec)
        norm = vec.norm()
        if norm < _ORTHO_VECTOR_TOL:
            warnings.warn('{0}: tried adding linearly dependent vector to ortho basis, discarding...'.format(self.__class__.__name__))
        else:
            self.add_vec(vec / norm)

    def project(self, u, return_coeffs=False):
        
        # Either this basis is orthonormal, or we've made the orthonormal basis...
        if self.is_orthonormal:
            c = self.dot(u)
            if return_coeffs:
                return self._colmul(c), c
            return self._columl(c)

        else:
            u_n = self.dot(u)
            try:
                if scipy.sparse.issparse(self.G):
                    y_n = scipy.sparse.linalg.spsolve(self.G, u_n)
                else:
                    y_n = scipy.linalg.solve(self.G, u_n, sym_pos=True)
            except np.linalg.LinAlgError as e:
                warnings.warn('{0}: linearly dependent with {1} vectors, projecting using SVD'.format(self.__class__.__name__, self.n))

                if np.any(self._vec_is_new):
                    if scipy.sparse.issparse(self.G):
                        self._U, self._S, self._V =  scipy.sparse.linalg.svds(self.G)
                    else:
                        self._U, self._S, self._V = np.linalg.svd(self.G)
                # This is the projection on the reduced rank basis 
                y_n = self._V.T @ ((self._U.T @ u_n) / self._S)

            # We allow the projection to be of the same type 
            # Also create it from the simple broadcast and sum (which surely should
            # be equivalent to some tensor product thing??)
            #u_p = type(self._vecs[0])((y_n * self.values_flat).sum(axis=2)) 
            
            if return_coeffs:
                return self._colmul(y_n), y_n

            return self._colmul(y_n)

    def orthonormalise(self):

        if self.n == 0:
            return self

        # We do a cholesky factorisation rather than a Gram Schmidt, as
        # we have a symmetric +ve definite matrix, so this is a cheap and
        # easy way to get an orthonormal basis from our previous basis
        if scipy.sparse.issparse(self.G):
            L = scipy.sparse.cholmod.cholesky(self.G)
        else:
            L = np.linalg.cholesky(self.G)
        
        self.L_inv = scipy.linalg.lapack.dtrtri(L.T)[0]
         
        #ortho_vecs = []
        #for i in range(self.n):
        #    ortho_vecs.append(self._colmul(self.L_inv[:,i]))
              
        #orthonormal_basis = type(self)(ortho_vecs)

        return self @ self.L_inv #orthonormal_basis

    ##
    ## TODO: Possibly deprecate!!
    ##

    def shuffle_vectors(self):
        random.shuffle(self._vecs)
        self.G = None

class BasisPair(object):
    """ This class automatically sets up the cross grammian, calculates
        beta, and can do the optimal reconstruction and calculate a favourable basis """

    def __init__(self, Wm, Vn, CG=None):

        self.Wm = Wm
        self.Vn = Vn
        
        self._Vn_vec_is_new = np.full(self.n, True)
        self._Wm_vec_is_new = np.full(self.n, True)
        if CG is not None:
            self._CG = CG
            self._Vn_vec_is_new = np.full(self.n, False)
            self._Wm_vec_is_new = np.full(self.n, False)

        self._U = self._S = self._V = None
    
    @property
    def n(self):
        return self.Vn.n
    @property
    def m(self):
        return self.Wm.n

    @property
    def CG(self):
        if np.any(self._Vn_vec_is_new) and np.any(self._Wm_vec_is_new):
            self._CG = self.Wm.H @ self.Vn
        return self._CG

    def add_Vn_vec(self, v):
        self.Vn.add_vec(v)
    
        if np.all(~self._Vn_vec_is_new) and np.all(~self._Wm_vec_is_new):
            self._CG = np.pad(self._CG, ((0,0),(0,1)), 'constant')

            for i in range(self.m):
                self._CG[i, -1] = self.Wm.vecs[i].dot(self.Vn.vecs[-1])

        self._U = self._V = self._S = None

    def add_Wm_vec(self, w):
        self.Wm.add_vec(w)

        if np.all(~self._Vn_vec_is_new) and np.all(~self._Wm_vec_is_new):
            self._CG = np.pad(self._CG, ((0,1),(0,0)), 'constant')
            for i in range(self.n):
                self._CG[-1, i] = self.Vn.vecs[i].dot(self.Wm.vecs[-1])

        self._U = self._V = self._S = None

    def __getitem__(self, idxs):
        if isinstance(idxs, (list, tuple)):
            if len(idxs) != 2:
                raise ValueError('{0}: requires two arguments for unlabelled subspace indexing')
            Wm_indices = idxs[0]
            Vn_indices = idxs[1]
        elif isinstance(idxs, dict):
            try:
                Wm_indices = idxs['Wm']
            except KeyError as e:
                Wm_indices = slice(0, self.m)
            try:
                Vn_indices = idxs['Vn']
            except KeyError as e:
                Vn_indices = slice(0, self.n)
        else:
            raise ValueError('{0}: unkown indexing format {1}'.format(self.__class__.__name__, idxs))

        sub = type(self)(self.Wm[Wm_indices], self.Vn[Vn_indices], CG=self._CG[Wm_indices, Vn_indices])        
        return sub
    
    def __setitem__(self, idxs, value):
        # Now this is confusing, probably need to make sure idx is a dict type
        # TODO: THIS MAKES NO SENSE, JUST REMOVE IT!
        if isinstance(idxs, dict):
            Wm_indices = None
            Vn_indices = None
            try:
                Wm_indices = idxs['Wm']
            except KeyError as e:
                pass
            try:
                Vn_indices = idxs['Vn']
            except KeyError as e:
                pass
        else:
            raise ValueError('{0}: requires labelled indexing for setitem'.format(self.__class__.__name__))
        
        if (Wm_indices is None and Vn_indices is None) or (Wm_indices is not None and Vn_indices is not None):
            raise ValueError('{0}: requires exactly one of \'Wm\' or \'Vn\' indices to be specified')
        else:
            if Wm_indices is not None:
                self.Wm[Wm_indices] = value
            if Vn_indices is not None:
                self.Vn[Vn_indices] = value

    def beta(self):
        if self.m < self.n:
            return 0.0
            
        if self._S is None:
            self._S = sp.linalg.svd(self.CG, compute_uv=False)

        return self.S[-1]

    def calc_svd(self):
        if not self.Wm.is_orthonormal or not self.Vn.is_orthonormal:
            raise Exception('Both Wm and Vn must be orthonormal to calculate meaningful SVD!')
        
        if self._U is None or self._S is None or self._V is None:
            self._U, self._S, self._V = sp.linalg.svd(self.CG)

    def Wm_singular_vec(self, index):
        self.calc_svd()
        return self.Wm @ self.U[:, index] 

    def Vn_singular_vec(self, index):
        self.calc_svd()
        return self.Vn @ self.V[index, :]

    def make_favorable_basis(self):
        if isinstance(self, FavorableBasisPair):
            return self
        
        self.calc_svd()
        
        fb = FavorableBasisPair(self.Wm @ self._U.T, #.ortho_matrix_multiply(self._U.T), 
                                self.Vn @ self._V, #.ortho_matrix_multiply(self._V),
                                CG=np.pad(np.diag(self._S), ((0,self.m-self.n), (0,0)), 'constant'))
        return fb

    def measure_and_reconstruct(self, u, disp_cond=False):
        """ Just a little helper function. Not sure we really want this here """ 
        u_p_W = self.Wm.dot(u)
        return self.optimal_reconstruction(u_p_W, disp_cond)

    def optimal_reconstruction(self, w, disp_cond=False):
        """ And here it is - the optimal reconstruction """
        if self.Vn.n > self.Wm.n:
            raise Exception('Error - Wm must be of higher dimensionality than Vn to be able to do optimal reconstruction')
        if not self.Wm.is_orthonormal or not self.Vn.is_orthonormal:
            raise Exception('Both Wm and Vn must be orthonormal to calculate the favourable basis!')
        try:
            c = scipy.linalg.solve(self.CG.T @ self.CG, self.CG.T @ w, sym_pos=True)
        except np.linalg.LinAlgError as e:
            print('Warning - unstable v* calculation, m={0}, n={1} for Wm and Vn, returning 0 function'.format(self.Wm.n, self.Vn.n))
            c = np.zeros(self.Vn.n)

        v_star = self.Vn @ c

        u_star = v_star + self.Wm @ (w - self.Wm.dot(v_star))

        # Note that W.project(v_star) = W.reconsrtuct(W.dot(v_star))
        # iff W is orthonormal...
        cond = np.linalg.cond(self.CG.T @ self.CG)
        if disp_cond:
            print('Condition number of G.T * G = {0}'.format(cond))
        
        return u_star, v_star, self.Wm @ w, self.Wm @ self.Wm.dot(v_star), cond

class FavorableBasisPair(BasisPair):
    """ This class automatically sets up the cross grammian, calculates
        beta, and can do the optimal reconstruction and calculated a favourable basis """

    def __init__(self, Wm, Vn, CG=None, S=None, U=None, V=None):
        # We quite naively assume that the basis we are given *is* in 
        # fact orthonormal, and don't do any testing...

        if S is not None:
            # Initialise with the Grammian equal to the singular values
            super().__init__(Wm, Vn, CG=CG)
            self._S = S
        else:
            super().__init__(Wm, Vn)
        if U is not None:
            self._U = U
        if V is not None:
            self._V = V

    def make_favorable_basis(self):
        return self

    def optimal_reconstruction(self, w, disp_cond=False):
        """ Optimal reconstruction is much easier with the favorable basis calculated 
            NB we have to assume that w is measured in terms of our basis Wn here... """
        
        w_tail = np.zeros(w.shape)
        w_tail[self.n:] = w[self.n:]
        
        v_star = self.Vn @ (w[:self.n] / self.S)
        u_star = v_star + self.Wm @ w_tail

        return u_star, v_star, self.Wm @ w, self.Wm @ self.Wm.dot(v_star)


