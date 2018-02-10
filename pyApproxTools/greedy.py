"""
greedy.py

Author: James Ashton Nichols
Start date: June 2017

The abstraction of some linear algebra in Hilbert spaces for doing functional analysis 
computations, where the class "Basis" does some predictable operations on the 
class "Vector", which has to have a dot product defined as well as typical linear algebra.

This submodule defines a variety of greedy algorithms, using the basis class
"""

import numpy as np
import copy
import time
import warnings

from pyApproxTools.vector import *
from pyApproxTools.pw_vector import *
from pyApproxTools.basis import *
from pyApproxTools.pw_basis import *

__all__ = ['CollectiveOMP', 'WorstCaseOMP', 'WorstVecOMP', 'GreedyApprox', 'MeasBasedGreedy', 'MeasBasedOMP', 'MeasBasedPP']

# This constant determines linear dependence in terms of max difference of norm (roughly sqrt machine tol)
_LD_ATOL = 1e-8


class GreedyMeasurement(object):

    def __init__(self, dictionary, Vn, Wm=None, verbose=False, remove=False):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        
        self.dictionary = copy.copy(dictionary)
        
        self.Vn = Vn
        self.Vn.make_grammian()
        self.Wm = Wm or Basis()
        self.Wm.make_grammian()
        self.m = self.Wm.n

        self.BP = None
        
        self.verbose = verbose
        self.remove = remove
        self.sel_crit = np.zeros(self.m)

    def initial_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        pass

    def next_step_choice(self, i):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        pass

    def construct_to_m(self, m_goal):

        if self.verbose:
            print('i \t || P_Vn (w - P_Wm w) ||')

        if self.Wm.n == 0:
            n0, crit = self.initial_choice()

            self.sel_crit = np.append(self.sel_crit, crit)
            
            self.Wm.add_vector(self.dictionary[n0])
            self.m = self.Wm.n
    
            if self.remove:
                del self.dictionary[n0]

        if self.BP is None:
            self.BP = BasisPair(self.Wm.orthonormalise(), self.Vn)
        elif self.BP.Wm is not self.Wm.orthonormal_basis or self.BP.Vn is not self.Vn:
            self.BP = BasisPair(self.Wm.orthonormalise(), self.Vn)
        
        while self.Wm.n < m_goal:
            
            ni, crit = self.next_step_choice(self.Wm.n)
            
            self.sel_crit = np.append(self.sel_crit, crit)

            self.BP.add_Wm_vector(self.dictionary[ni])
            self.Wm.add_vector(self.dictionary[ni])
            self.m = self.Wm.n

            if self.remove:
                del self.dictionary[ni]
                   
        if self.verbose:
            print('\n\nDone!')
        
        return self.Wm

    def construct_to_beta(self, beta_goal, m_max_ratio=10):

        if self.verbose:
            print('i \t || P_Vn (w - P_Wm w) ||')

        if self.Wm.n == 0:
            n0, crit = self.initial_choice()
            self.sel_crit = np.append(self.sel_crit, crit)
                
            self.Wm.add_vector(self.dictionary[n0])
            self.m = self.Wm.n
        
            if self.remove:
                del self.dictionary[n0]

        if self.BP is None:
            self.BP = BasisPair(self.Wm.orthonormalise(), self.Vn)
        elif self.BP.Wm is not self.Wm.orthonormal_basis or self.BP.Vn is not self.Vn:
            self.BP = BasisPair(self.Wm.orthonormalise(), self.Vn)
        
        while self.BP.beta() < beta_goal:
            ni, crit = self.next_step_choice(self.Wm.n)
            
            self.sel_crit = np.append(self.sel_crit, crit)
            
            self.BP.add_Wm_vector(self.dictionary[ni])
            self.Wm.add_vector(self.dictionary[ni])

            self.m = self.Wm.n

            if self.remove:
                del self.dictionary[ni]
            
            if self.Wm.n > m_max_ratio * self.Vn.n:
                print('Ceiling reached for Wm size before beta_goal reached!')
                break

        if self.verbose:
            print('\n\nDone!')
        
        return self.Wm

class CollectiveOMP(GreedyMeasurement):
    """ Probably should rename this class, but it implements the Collective OMP algorithm for constructing Wm """

    def __init__(self, dictionary, Vn, Wm=None, verbose=False, remove=False):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        super().__init__(dictionary, Vn, Wm=Wm, verbose=verbose, remove=remove)
       
    def initial_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
       
        norms = np.zeros(len(self.dictionary))
        for i in range(len(self.dictionary)):
            for phi in self.Vn.vecs:
                norms[i] += phi.dot(self.dictionary[i]) ** 2
        
        n0 = np.argmax(norms)

        return n0, norms[n0]

    def next_step_choice(self, i):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
 
        next_crit = np.zeros(len(self.dictionary))
        # We go through the dictionary and find the max of || f ||^2 - || P_Vn f ||^2
        for phi in self.Vn.vecs:
            phi_perp = phi - self.Wm.project(phi)
            for j in range(len(self.dictionary)):
                next_crit[j] += phi_perp.dot(self.dictionary[j]) ** 2
                #p_V_d[i] = self.Wm.project(self.dictionary[i]).norm()

        ni = np.argmax(next_crit)
        
        if self.verbose:
            print('{0} : \t {1} \t {2}'.format(i, ni, next_crit[ni]))

        return ni, next_crit[ni]

class WorstCaseOMP(GreedyMeasurement):
    """ Now the slightly simpler (to analyse) parallel OMP that looks at Vn vecs individually """

    def __init__(self, dictionary, Vn, Wm=None, verbose=False, remove=False):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        super().__init__(dictionary, Vn, Wm=Wm, verbose=verbose, remove=remove)
        
        self.BP = None
        self.Vtilde = []

    def initial_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        v0 = self.Vn.vecs[0]

        dots = np.zeros(len(self.dictionary))
        for i, d in enumerate(self.dictionary):
            dots[i] = v0.dot(d)
        
        n0 = np.argmax(dots)
      
        self.Vtilde.append(v0)

        return n0, dots[n0]

    def next_step_choice(self, i):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        if self.BP.Wm is not self.Wm.orthonormal_basis or self.BP.Vn is not self.Vn:
            self.BP = BasisPair(self.Wm.orthonormalise(), self.Vn)
        
        next_crit = np.zeros(len(self.dictionary))
 
        # We go through the dictionary and find the max of || f ||^2 - || P_Vn f ||^2
        v = self.BP.Vn_singular_vec(-1)
        v_perp = v - self.Wm.project(v)
        for j in range(len(self.dictionary)):
            next_crit[j] = abs(v_perp.dot(self.dictionary[j]))
         
        ni = np.argmax(next_crit)
        self.Vtilde.append(v)
        
        if self.verbose:
            print('{0} : \t {1} \t {2}'.format(i, ni, next_crit[ni]))

        return ni, next_crit[ni]

class WorstVecOMP(CollectiveOMP):
    """ Now we look at the worst of the basis vectors instead of over the whole space.. hopefully easier to
        analyse and prove, and faster to do... """

    def __init__(self, m, dictionary, Vn, Wm=None, verbose=False, remove=True):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        super().__init__(m, dictionary, Vn, Wm=Wm, verbose=verbose, remove=remove)
            
    def initial_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        v0 = self.Vn.vecs[0]

        dots = np.zeros(len(self.dictionary))
        for i in range(len(self.dictionary)):
            dots[i] = v0.dot(self.dictionary[i])

        n0 = np.argmax(dots)
      
        return n0, dots[n0]

    def next_step_choice(self, i):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        next_crit = np.zeros(len(self.dictionary))
        
        phi_perps = np.zeros(self.Vn.n)
        # First we find the phi_j that has the largest phi_j - P_Wm phi_j
        for j in range(self.Vn.n):
            phi_perps[j] = (self.Vn.vecs[j] - self.Wm.project(self.Vn.vecs[j])).norm()

        # This corresponds with vector with the smallest singular value from the SVD
        phi = self.Vn.vecs[phi_perps.argmin()]

        phi_perp = phi - self.Wm.project(phi)
        for j in range(len(self.dictionary)):
            next_crit[j] = abs(phi_perp.dot(self.dictionary[j]))
        
        ni = np.argmax(next_crit)

        if self.verbose:
            print('{0} : \t {1}'.format(i, next_crit[ni]))

        return ni, next_crit[ni]




"""

These classes are for greedy construction of Vn given a particular measured u in a particular basis Wm. 
One feature is the use of only the R^(n x m) representation of everything, saving the time of the inner
product calculations in V_h or V, the ambient solution space

"""
class LinearlyDependent(Exception): pass

class GreedyApprox(object):

    def __init__(self, dictionary, Vn=None, verbose=False, remove=False):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        
        self.dictionary = copy.copy(dictionary)
        
        self.Vn = Vn or Basis()
        self.Vn.make_grammian()
       
        self.verbose = verbose
        self.remove = remove
        self.sel_crit = np.array([])
        self.dict_sel = np.array([], dtype=np.int32)

    @property
    def n(self):
        return self.Vn.n
    
    def initial_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        self.norms = np.zeros(len(self.dictionary))
        for j, v in enumerate(self.dictionary):
            self.norms[j] = v.norm()

        n0 = np.argmax(self.norms)
        crit = self.norms[n0]

        self.Vn.add_vector(self.dictionary[n0])

        return n0, crit
 

    def next_step_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
    
        p_V_d = np.zeros(len(self.dictionary))
        # We go through the dictionary and find the max of perp
        for j, v in enumerate(self.dictionary):
            v_perp = v - self.Vn.project(v)
            p_V_d[j] = v_perp.norm()
        
        if np.all(np.isclose(p_V_d, 0.0, atol=_LD_ATOL)):
            raise LinearlyDependent()
        
        ni = np.argmax(p_V_d)
        crit = p_V_d[ni]

        self.Vn.add_vector(self.dictionary[ni])
        
        # Test linear indpendence
        #lambdas = np.linalg.eigvalsh(self.Vn.G)
        #if np.any(np.isclose(lambdas, 0.0, atol=1e-10)):
        #    raise LinearlyDependent()
    
        return ni, crit
    
    def construct_to_n(self, n_goal):
         
        if self.verbose:
            print('i \t Selection \t Sel. criteria')
        
        if self.Vn.n == 0:
            n0, crit = self.initial_choice()

            self.dict_sel = np.append(self.dict_sel, n0) 
            self.sel_crit = np.append(self.sel_crit, crit)

            if self.remove:
                del self.dictionary[n0]
            
            if self.verbose:
                print('{0} : \t {1} \t {2}'.format(self.Vn.n, n0, crit))

        try: 
            while self.Vn.n < n_goal:
                
                ni, crit = self.next_step_choice()
                
                self.dict_sel = np.append(self.dict_sel, ni) 
                self.sel_crit = np.append(self.sel_crit, crit)

                if self.remove:
                    del self.dictionary[ni]
                
                if self.verbose:
                    print('{0} : \t {1} \t\t {2}'.format(self.Vn.n, ni, crit))

        except LinearlyDependent:
            print('Vn spans all dictionary points at n={0}, stopping greedy'.format(self.n))

        if self.verbose:
            print('Done!')
        
        return self.Vn

class MeasBasedGreedy(GreedyApprox):
    """ Measurement based greedy algorithm """ 
    def __init__(self, dictionary, u, Wm, Vn=None, verbose=False, remove=False):
 
        if not Wm.is_orthonormal:
            raise Exception('Need orthonormal Wm for greedy approx construction')

        self.Wm = Wm
        self.w = self.Wm.project(u)

        self.BP = None
        self.beta = np.zeros(self.m)

        super().__init__(dictionary, Vn=Vn, verbose=verbose, remove=remove)

        self.Wdict = np.zeros((len(dictionary), self.m))
        for i, v in enumerate(dictionary):
            self.Wdict[i, :] = self.Wm.dot(v)

    @property
    def m(self):
        return self.Wm.n
    
    def reset_u(self, u):
        if self.remove:
            warnings.warn('Resetting greedy constructor with dictionary removal - \
                           dictionary is of length {0}'.format(len(self.dictionary)))

        # This is to reset with a new u but same dictionary dots, save lots of time...
        self.w = self.Wm.project(u)
        self.Vn = type(self.Vn)()
        self.Vn.make_grammian()
        self.BP = None
        self.beta = np.zeros(self.m)

        self.sel_crit = np.array([])
        self.dict_sel = np.array([], dtype=np.int32)

    def construct_to_n(self, n_goal):

        self.beta.resize(n_goal)
        super().construct_to_n(n_goal)

        return self.Vn

    def initial_choice(self):

        self.norms = np.linalg.norm(self.Wdict, axis=1)

        n0 = np.argmax(self.norms)
        crit = self.norms[n0]

        self.Vn.add_vector(self.dictionary[n0])

        if self.BP is None or self.BP.Vn is not self.Vn.orthonormal_basis:
            self.BP = BasisPair(self.Wm, self.Vn.orthonormalise())
        self.beta[self.n - 1] = self.BP.beta()
    
        if self.remove:
            self.Wdict = np.delete(self.Wdict, (n0), axis=0)

        return n0, crit

    def next_step_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
 
        p_V_d = np.zeros(len(self.dictionary))
        
        w_perp = self.w - self.Vn.project(self.w)
        # We go through the dictionary and find the max of 
        for j, v in enumerate(self.dictionary):
            p_V_d[j] = np.abs(w_perp.dot(v))
        
        if np.all(np.isclose(p_V_d, 0.0, atol=_LD_ATOL)):
            raise LinearlyDependent()

        ni = np.argmax(p_V_d)
        crit = p_V_d[ni]
        
        self.Vn.add_vector(self.dictionary[ni])
        
        # Test linear indpendence
        #lambdas = np.linalg.eigvalsh(self.Vn.G)
        #if np.any(np.isclose(lambdas, 0.0, atol=1e-10)):
        #    raise LinearlyDependent()
 
        self.BP.add_Vn_vector(self.dictionary[ni])
        self.beta[self.n-1] = self.BP.beta()

        if self.remove:
            self.Wdict = np.delete(self.Wdict, (ni), axis=0)

        return ni, crit

class MeasBasedOMP(GreedyApprox):
    """ Measurement based greedy orthogonal matching pursuit """ 

    def __init__(self, dictionary, u, Wm, Vn=None, verbose=False, remove=False):
 
        if not Wm.is_orthonormal:
            raise Exception('Need orthonormal Wm for greedy approx construction')

        self.Wm = Wm
        self.w, self.w_coeffs = self.Wm.project(u, return_coeffs=True)

        self.BP = None
        self.beta = np.zeros(self.m)

        super().__init__(dictionary, Vn=Vn, verbose=verbose, remove=remove)

        self.Wdict = np.zeros((len(dictionary), self.m))
        for i, v in enumerate(dictionary):
            self.Wdict[i, :] = self.Wm.dot(v)
            self.Wdict[i, :] /= np.linalg.norm(self.Wdict[i,:]) # NOTE - Should normalise here

        self.Zn = None

    @property
    def m(self):
        return self.Wm.n
 
    def reset_u(self, u):
        if self.remove:
            warnings.warn('Resetting greedy constructor with dictionary removal - \
                           dictionary is of length {0}'.format(len(self.dictionary)))

        # This is to reset with a new u but same dictionary dots, save lots of time...
        self.w, self.w_coeffs = self.Wm.project(u, return_coeffs=True)
        self.Vn = type(self.Vn)()
        self.Vn.make_grammian()
        self.BP = None
        self.beta = np.zeros(self.m)

        self.sel_crit = np.array([])
        self.dict_sel = np.array([], dtype=np.int32)

    def construct_to_n(self, n_goal):

        self.beta.resize(n_goal)
        super().construct_to_n(n_goal)

        return self.Vn

    def initial_choice(self):

        p_V_d = np.abs(np.dot(self.w_coeffs, self.Wdict.T))
        n0 = np.argmax(p_V_d)
        crit = p_V_d[n0]

        self.Vn.add_vector(self.dictionary[n0])
        self.Zn = self.Wdict[n0,:][:,np.newaxis]

        if self.BP is None or self.BP.Vn is not self.Vn.orthonormal_basis:
            self.BP = BasisPair(self.Wm, self.Vn.orthonormalise())
        self.beta[self.n - 1] = self.BP.beta()
    
        if self.remove:
            self.Wdict = np.delete(self.Wdict, (n0), axis=0)

        return n0, crit

    def next_step_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        w_perp =  self.w_coeffs - np.linalg.lstsq(self.Zn, self.w_coeffs, rcond=None)[0] @ self.Zn.T
        
        p_V_d = np.abs(np.dot(w_perp, self.Wdict.T)) #self.Wm.dot(v_perp)))

        if np.all(np.isclose(p_V_d, 0.0, atol=_LD_ATOL)):
            raise LinearlyDependent()

        ni = np.argmax(p_V_d)
        crit = p_V_d[ni]

        self.Vn.add_vector(self.dictionary[ni])
        self.Zn = np.hstack((self.Zn, self.Wdict[ni, :][:,np.newaxis]))
        
        self.BP.add_Vn_vector(self.dictionary[ni])
        self.beta[self.n-1] = self.BP.beta()

        if self.remove:
            self.Wdict = np.delete(self.Wdict, (ni), axis=0)

        return ni, crit

class MeasBasedPP(GreedyApprox):
    """ Measurement based greedy orthogonal matching pursuit """ 

    def __init__(self, dictionary, u, Wm, Vn=None, verbose=False, remove=False):
 
        if not Wm.is_orthonormal:
            raise Exception('Need orthonormal Wm for greedy approx construction')

        self.Wm = Wm
        self.w, self.w_coeffs = self.Wm.project(u, return_coeffs=True) 

        self.BP = None
        self.beta = np.zeros(self.m)

        super().__init__(dictionary, Vn=Vn, verbose=verbose, remove=remove)

        self.Wdict = np.zeros((len(dictionary), self.m))
        for i, v in enumerate(dictionary):
            self.Wdict[i, :] = self.Wm.dot(v)
            self.Wdict[i, :] /= np.linalg.norm(self.Wdict[i,:]) # NOTE - Should normalise here

        self.Zn = None

    @property
    def m(self):
        return self.Wm.n
 
    def reset_u(self, u):
        if self.remove:
            warnings.warn('Resetting greedy constructor with dictionary removal - \
                           dictionary is of length {0}'.format(len(self.dictionary)))

        # This is to reset with a new u but same dictionary dots, save lots of time...
        self.w, self.w_coeffs = self.Wm.project(u, return_coeffs=True)
        self.Vn = type(self.Vn)()
        self.Vn.make_grammian()
        self.BP = None
        self.beta = np.zeros(self.m)

        self.sel_crit = np.array([])
        self.dict_sel = np.array([], dtype=np.int32)

    def construct_to_n(self, n_goal):

        self.beta.resize(n_goal)
        super().construct_to_n(n_goal)

        return self.Vn

    def initial_choice(self):

        p_V_d = np.abs(np.dot(self.w_coeffs, self.Wdict.T))
        n0 = np.argmax(p_V_d)
        crit = p_V_d[n0]

        self.Vn.add_vector(self.dictionary[n0])
        self.Zn = self.Wdict[n0,:][:,np.newaxis]

        if self.BP is None or self.BP.Vn is not self.Vn.orthonormal_basis:
            self.BP = BasisPair(self.Wm, self.Vn.orthonormalise())
        self.beta[self.n - 1] = self.BP.beta()
    
        if self.remove:
            self.Wdict = np.delete(self.Wdict, (n0), axis=0)

        return n0, crit

    def next_step_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        p_V_d = np.zeros(len(self.dictionary))
        
        for j, z in enumerate(self.Wdict):
            Zn_ext =  np.hstack((self.Zn, z[:,np.newaxis]))
            w_perp = self.w_coeffs - np.linalg.lstsq(Zn_ext, self.w_coeffs, rcond=None)[0] @ Zn_ext.T
            p_V_d[j] = np.linalg.norm(w_perp)
       
        if np.any(np.isclose(p_V_d, 0.0, atol=_LD_ATOL)):
            raise LinearlyDependent()

        ni = np.argmin(p_V_d)
        crit = p_V_d[ni]

        self.Vn.add_vector(self.dictionary[ni])
        self.Zn = np.hstack((self.Zn, self.Wdict[ni, :][:,np.newaxis]))
        
        self.BP.add_Vn_vector(self.dictionary[ni])
        self.beta[self.n-1] = self.BP.beta()

        if self.remove:
            self.Wdict = np.delete(self.Wdict, (ni), axis=0)

        return ni, crit


