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
import pdb

from pyApproxTools.vector import *
from pyApproxTools.basis import *

__all__ = ['CollectiveOMP', 'WorstCaseOMP', 'WorstVecOMP']

class CollectiveOMP(object):
    """ Probably should rename this class, but it implements the Collective OMP algorithm for constructing Wm """

    def __init__(self, m, dictionary, Vn, Wm=None, verbose=False, remove=True):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        
        self.dictionary = copy.copy(dictionary)

        self.m = m
        self.Vn = Vn
        self.Wm = Wm or Basis()

        self.verbose = verbose
        self.remove = remove
        self.sel_crit = np.zeros(m)

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
            print('{0} : \t {1}'.format(i, next_crit[ni]))

        return ni, next_crit[ni]

    def construct_basis(self):
        " The construction method should be generic enough to support all variants of the greedy algorithms """
        
        if self.Wm is None:
            n0, self.sel_crit[0] = self.initial_choice()
             
            self.Wm = Basis([self.dictionary[n0]])
            self.Wm.make_grammian()
 
            if self.remove:
                del self.dictionary[n0]

            if self.verbose:
                print('\n\nGenerating basis from greedy algorithm with dictionary: ')
                print('i \t || P_Vn (w - P_Wm w) ||')

            for i in range(1, self.m):
                
                ni, self.sel_crit[i] = self.next_step_choice(i)
                   
                self.Wm.add_vector(self.dictionary[ni], incr_ortho=True)
 
                if self.remove:
                    del self.dictionary[ni]
                       
            if self.verbose:
                print('\n\nDone!')
        else:
            print('Greedy basis already computed!')
        
        return self.Wm


class WorstCaseOMP(CollectiveOMP):
    """ Now the slightly simpler (to analyse) parallel OMP that looks at Vn vecs individually """

    def __init__(self, m, dictionary, Vn, Wm=None, verbose=False, remove=True):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        super().__init__(m, dictionary, Vn, Wm=Wm, verbose=verbose, remove=remove)
            
        self.BP = None
        self.Vtilde = []

    def initial_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        v0 = self.Vn.vecs[0]

        dots = np.zeros(len(self.dictionary))
        for i in range(len(self.dictionary)):
            dots[i] = v0.dot(self.dictionary[i])

        n0 = np.argmax(dots)
      
        self.Vtilde.append(v0)

        return n0, dots[n0]

    def next_step_choice(self, i):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        next_crit = np.zeros(len(self.dictionary))
        # We go through the dictionary and find the max of || f ||^2 - || P_Vn f ||^2
        BP = BasisPair(self.Wm.orthonormalise(), self.Vn)
          
        v = BP.Vn_singular_vec(-1)
        
        v_perp = v - self.Wm.project(v)
        for j in range(len(self.dictionary)):
            next_crit[j] = abs(v_perp.dot(self.dictionary[j]))
        
        ni = np.argmax(next_crit)
        self.Vtilde.append(v)

        if self.verbose:
            print('{0} : \t {1}'.format(i, next_crit[ni]))

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

