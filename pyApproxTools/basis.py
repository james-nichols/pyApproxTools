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

class Basis(object):
    """ Class representing the mathematical concept of a Basis. Routines available
    include:

    subspace(indices)
    subspace_mask(mask)
    make_grammian()
    cross_grammian(other)
    project(u)
    reconstruct(c)
    matrix_multiply(A)
    orthonormalise()
    """

    def __init__(self, vecs=None, space='H1', is_orthonormal=False):
        
        self.vecs = vecs or []
        
        self.space = space

        self.is_orthonormal = is_orthonormal
        if is_orthonormal:
            self.orthonormal_basis = self
            self.G = np.eye(self.n)
        else:
            self.orthonormal_basis = None
            self.G = None
        
        self.L_inv = None
        self.U = self.S = self.V = None
    

    @property
    def n(self):
        return len(self.vecs)

    def add_vector(self, vec, incr_ortho=False, check_ortho=True):
        """ Add just one vector, so as to make the new Grammian calculation quick """
        
        vec = copy.deepcopy(vec)
        if self.is_orthonormal:    
            """ add a vector - if it is orthonormal already just add it, other wise do one gram-schmidt step """
            v_dot = np.zeros(self.n)
            if check_ortho:
                for i, v in enumerate(self.vecs):
                    v_dot[i] = v.dot(vec)

            if any(np.abs(v_dot) > 1e-13):
                # We do a Gram-Schmidt style removal
                for i, v in enumerate(self.vecs):
                    vec = vec - v_dot[i] * v
                n = vec.norm()
                if n < 1e-13:
                    warnings.warn('{0}: tried adding linearly dependent vector to ortho basis, discarding...'.format(self.__class__.__name__))
                else:
                    self.vecs.append(vec / n)
            else:
                self.vecs.append(vec/vec.norm())
   
            if self.G is not None:
                self.G = np.eye(self.n)
        else:
            self.vecs.append(vec)

            if self.G is not None:
                self.G = np.pad(self.G, ((0,1),(0,1)), 'constant')
                for i in range(self.n):
                    self.G[self.n-1, i] = self.G[i, self.n-1] = self.vecs[-1].dot(self.vecs[i])

                # This is for performance's sake, if we only add one vec then we can also
                # increment the othogonal system associated
                if incr_ortho and self.orthonormal_basis is not None:
                    self.orthonormal_basis.add_vector(vec) 
                else:
                    # The new basis means the previous basis is now 
                    self.orthonormal_basis = None

        # Unfortunately there's no incremental SVD solution that I know of...
        self.U = self.V = self.S = None

    def shuffle_vectors(self):
        random.shuffle(self.vecs)
        self.G = None

    def subspace(self, indices):
        """ Select a subspace corresponding to a subset of the basis, where indices is a Slice object """
        sub = type(self)(self.vecs[indices], space=self.space, is_orthonormal=self.is_orthonormal)
        
        if self.G is not None:
            sub.G = self.G[indices, indices]

        return sub

    def subspace_mask(self, mask):
        """ Select a subspace corresponding to a subset of the basis, where indices is a binary mask """
        if mask.shape[0] != len(self.vecs):
            raise Exception('Subspace mask must be the same size as length of vectors')

        sub = type(self)(list(itertools.compress(self.vecs, mask)), space=self.space, is_orthonormal=self.is_orthonormal)
        if self.G is not None:
            sub.G = self.G[mask,mask]
        return sub

    def dot(self, u):
        u_d = np.zeros(self.n)
        for i, v in enumerate(self.vecs):
            u_d[i] = v.dot(u)
        return u_d

    def make_grammian(self):
        if self.G is None:
            self.G = np.zeros([self.n,self.n])
            for i in range(self.n):
                for j in range(i, self.n):
                    self.G[i,j] = self.G[j,i] = self.vecs[i].dot(self.vecs[j])

    def cross_grammian(self, other):
        
        if other.space != self.space:
            raise Exception('Bases not in the same space!')

        CG = np.zeros([self.n, other.n])
        
        for i in range(self.n):
            for j in range(other.n):
                CG[i,j] = self.vecs[i].dot(other.vecs[j])
        return CG

    def project(self, u, return_coeffs=False):
        
        # Either this basis is orthonormal, or we've made the orthonormal basis...
        if self.is_orthonormal:
            if return_coeffs:
                c = self.dot(u)
                return self.reconstruct(c), c
            return self.reconstruct(self.dot(u))
        elif self.orthonormal_basis is not None:
            return self.orthonormal_basis.project(u)
        else:
            if self.G is None:
                self.make_grammian()

            u_n = self.dot(u)
            try:
                if scipy.sparse.issparse(self.G):
                    y_n = scipy.sparse.linalg.spsolve(self.G, u_n)
                else:
                    y_n = scipy.linalg.solve(self.G, u_n, sym_pos=True)
            except np.linalg.LinAlgError as e:
                print('Warning - basis is linearly dependent with {0} vectors, projecting using SVD'.format(self.n))

                if self.U is None:
                    if scipy.sparse.issparse(self.G):
                        self.U, self.S, self.V =  scipy.sparse.linalg.svds(self.G)
                    else:
                        self.U, self.S, self.V = np.linalg.svd(self.G)
                # This is the projection on the reduced rank basis 
                y_n = self.V.T @ ((self.U.T @ u_n) / self.S)

            # We allow the projection to be of the same type 
            # Also create it from the simple broadcast and sum (which surely should
            # be equivalent to some tensor product thing??)
            #u_p = type(self.vecs[0])((y_n * self.values_flat).sum(axis=2)) 
            
            if return_coeffs:
                return self.reconstruct(y_n), y_n

            return self.reconstruct(y_n)

    def reconstruct(self, c):
        # Build a function from a vector of coefficients
        if len(c) != len(self.vecs):
            raise Exception('Coefficients and vectors must be of same length!')
         
        u_p = type(self.vecs[0])()
        for i, c_i in enumerate(c):
            if c_i != 0:
                u_p += c_i * self.vecs[i] 
        return u_p

    def matrix_multiply(self, M):
        # Build another basis from a matrix, essentially just calls 
        # reconstruct for each row in M
        if M.shape[1] != self.n:
            raise Exception('M must have {0} cols'.format(self.n))

        vecs = []
        for i in range(M.shape[0]):
            vecs.append(self.reconstruct(M[i,:]))
        
        return type(self)(vecs, space=self.space)

    def ortho_matrix_multiply(self, M):
        # Build another basis from an orthonormal matrix, 
        # which means that the basis that comes from it
        # is also orthonormal *if* it was orthonormal to begin with
        if M.shape[0] != M.shape[1] or M.shape[1] != self.n:
            raise Exception('M must be a {0}x{1} square matrix'.format(self.n, self.n))

        vecs = []
        for i in range(M.shape[0]):
            vecs.append(self.reconstruct(M[i,:]))
        
        # In case this is an orthonormal basis
        return type(self)(vecs, space=self.space, is_orthonormal=True)

    def orthonormalise(self):

        if self.n == 0:
            return self

        if self.orthonormal_basis is None or self.orthonormal_basis.n != self.n:
            if self.G is None:
                self.make_grammian()
           
            # We do a cholesky factorisation rather than a Gram Schmidt, as
            # we have a symmetric +ve definite matrix, so this is a cheap and
            # easy way to get an orthonormal basis from our previous basis
            if scipy.sparse.issparse(self.G):
                L = scipy.sparse.cholmod.cholesky(self.G)
            else:
                L = np.linalg.cholesky(self.G)
            self.L_inv = scipy.linalg.lapack.dtrtri(L.T)[0]
             
            ortho_vecs = []
            for i in range(self.n):
                ortho_vecs.append(self.reconstruct(self.L_inv[:,i]))
                  
            self.orthonormal_basis = type(self)(ortho_vecs, space=self.space, is_orthonormal=True)

        return self.orthonormal_basis

class BasisPair(object):
    """ This class automatically sets up the cross grammian, calculates
        beta, and can do the optimal reconstruction and calculate a favourable basis """

    def __init__(self, Wm, Vn, CG=None):

        if Vn.space != Wm.space:
            raise Exception('Error - Wm and Vn must be in the same space')

        self.Wm = Wm
        self.Vn = Vn
        
        if CG is not None:
            self.CG = CG
        else:
            self.CG = self.cross_grammian()

        self.U = self.S = self.V = None
    
    @property
    def n(self):
        return self.Vn.n
    @property
    def m(self):
        return self.Wm.n

    def cross_grammian(self):
        CG = np.zeros([self.m, self.n])
        
        for i in range(self.m):
            for j in range(self.n):
                CG[i,j] = self.Wm.vecs[i].dot(self.Vn.vecs[j])
        return CG
    
    def add_Vn_vector(self, v):
        self.Vn.add_vector(v)

        if self.CG is not None:
            self.CG = np.pad(self.CG, ((0,0),(0,1)), 'constant')

            for i in range(self.m):
                self.CG[i, -1] = self.Wm.vecs[i].dot(self.Vn.vecs[-1])

        self.U = self.V = self.S = None

    def add_Wm_vector(self, w):
        self.Wm.add_vector(w)

        if self.CG is not None:
            self.CG = np.pad(self.CG, ((0,1),(0,0)), 'constant')
            for i in range(self.n):
                self.CG[-1, i] = self.Vn.vecs[i].dot(self.Wm.vecs[-1])

        self.U = self.V = self.S = None

    def subspace(self, Wm_indices=None, Vn_indices=None):
        if Wm_indices is None:
            Wm_indices = slice(0, self.m)
        if Vn_indices is None:
            Vn_indices = slice(0, self.n)
        sub = type(self)(self.Wm.subspace(Wm_indices), self.Vn.subspace(Vn_indices), CG=self.CG[Wm_indices, Vn_indices])        
        return sub

    def subspace_mask(self, Wm_mask=None, Vn_mask=None):
        if Wm_mask is None:
            Wm_mask = np.ones(self.m, dtype=np.bool)
        if Vn_mask is None:
            Vn_mask = np.ones(self.n, dtype=np.bool)
        
        if Wm_mask.shape[0] != self.m and Vn_mask.shape[0] != self.n:
            raise Exception('Subspace mask must be the same size as length of vectors')

        sub = type(self)(self.Wm.subspace(mask), self.Vn.subspace(mask), CG=self.CG[Wm_mask, Vn_mask])
        return sub

    def beta(self):
        if self.Wm.n < self.Vn.n:
            return 0.0
            
        if self.U is None or self.S is None or self.V is None:
            self.calc_svd()

        return self.S[-1]

    def calc_svd(self):
        if self.U is None or self.S is None or self.V is None:
            self.U, self.S, self.V = np.linalg.svd(self.CG)

    def Wm_singular_vec(self, index):
        if not self.Wm.is_orthonormal or not self.Vn.is_orthonormal:
            raise Exception('Both Wm and Vn must be orthonormal to calculate the largest singular vec!')
        if self.U is None or self.S is None or self.V is None:
            self.calc_svd()

        return self.Wm.reconstruct(self.U[:, index])

    def Vn_singular_vec(self, index):
        if not self.Wm.is_orthonormal or not self.Vn.is_orthonormal:
            raise Exception('Both Wm and Vn must be orthonormal to calculate the largest singular vec!')
        if self.U is None or self.S is None or self.V is None:
            self.calc_svd()

        return self.Vn.reconstruct(self.V[index, :])

    def make_favorable_basis(self):
        if isinstance(self, FavorableBasisPair):
            return self
        
        if not self.Wm.is_orthonormal or not self.Vn.is_orthonormal:
            raise Exception('Both Wm and Vn must be orthonormal to calculate the favourable basis!')

        if self.U is None or self.S is None or self.V is None:
            self.calc_svd()

        fb = FavorableBasisPair(self.Wm.ortho_matrix_multiply(self.U.T), 
                                self.Vn.ortho_matrix_multiply(self.V),
                                S=self.S, U=np.eye(self.n), V=np.eye(self.m))
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

        v_star = self.Vn.reconstruct(c)

        u_star = v_star + self.Wm.reconstruct(w - self.Wm.dot(v_star))

        # Note that W.project(v_star) = W.reconsrtuct(W.dot(v_star))
        # iff W is orthonormal...
        cond = np.linalg.cond(self.CG.T @ self.CG)
        if disp_cond:
            print('Condition number of G.T * G = {0}'.format(cond))
        
        return u_star, v_star, self.Wm.reconstruct(w), self.Wm.reconstruct(self.Wm.dot(v_star)), cond

class FavorableBasisPair(BasisPair):
    """ This class automatically sets up the cross grammian, calculates
        beta, and can do the optimal reconstruction and calculated a favourable basis """

    def __init__(self, Wm, Vn, S=None, U=None, V=None):
        # We quite naively assume that the basis we are given *is* in 
        # fact orthonormal, and don't do any testing...

        if S is not None:
            # Initialise with the Grammian equal to the singular values
            super().__init__(Wm, Vn, CG=S)
            self.S = S
        else:
            super().__init__(Wm, Vn)
        if U is not None:
            self.U = U
        if V is not None:
            self.V = V

    def make_favorable_basis(self):
        return self

    def optimal_reconstruction(self, w, disp_cond=False):
        """ Optimal reconstruction is much easier with the favorable basis calculated 
            NB we have to assume that w is measured in terms of our basis Wn here... """
        
        w_tail = np.zeros(w.shape)
        w_tail[self.n:] = w[self.n:]
        
        v_star = self.Vn.reconstruct(w[:self.n] / self.S)
        u_star = v_star + self.Wm.reconstruct(w_tail)

        return u_star, v_star, self.Wm.reconstruct(w), self.Wm.reconstruct(self.Wm.dot(v_star))


