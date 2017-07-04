"""
pw_utils.py

Author: James Ashton Nichols
Start date: June 2017

Some utils including a first implementation of the FEM scheme
"""

import math
import numpy as np
import scipy.linalg
import scipy.sparse
from itertools import *
import copy

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import axes3d, Axes3D

from pyApproxTools.vector import *
from pyApproxTools.basis import *
from pyApproxTools.pw_vector import *
from pyApproxTools.pw_basis import *

__all__ = ['DyadicFEMSolver','make_pw_hat_basis','make_pw_sin_basis','make_pw_reduced_basis','make_pw_random_local_integration_basis','make_local_integration_basis']

class DyadicFEMSolver(object):
    """ Solves the -div( a nabla u ) = f PDE on a grid, with a given by 
        some random / deterministic field, with dirichelet boundary conditions """

    def __init__(self, div, rand_field, f):
       
        self.div = div
        # -1 as we are interested in the grid points, of which there is an odd number
        self.n_side = 2**self.div - 1
        self.n_el = self.n_side * self.n_side
        self.h = 1.0 / (self.n_side + 1)

        # Makes an appropriate sized field for our FEM grid
        a = rand_field.interpolate(self.div).values
        
        # Now we make the various diagonals
        diag = 2.0 * (a[:-1, :-1] + a[:-1,1:] + a[1:,:-1] + a[1:, 1:]).flatten()
        
        # min_diag is below the diagonal, hence deals with element to the left in the FEM grid
        lr_diag = -(a[1:, 1:] + a[:-1, 1:]).flatten()
        lr_diag[self.n_side-1::self.n_side] = 0 # These corresponds to edges on left or right extreme
        lr_diag = lr_diag[:-1]
        
        # Far min deals with the element that is above
        ud_diag = -(a[1:-1, 1:] + a[1:-1, :-1]).flatten()

        self.A = scipy.sparse.diags([diag, lr_diag, lr_diag, ud_diag, ud_diag], [0, -1, 1, -self.n_side, self.n_side]).tocsr()
        self.f = f * 0.5 * self.h * self.h * np.ones(self.n_el)

        self.u = PWLinearSqDyadicH1(np.zeros([self.n_side + 2, self.n_side + 2]), self.div)

    def solve(self):
        """ The bilinear form simply becomes \int_D a nab u . nab v = \int_D f """
        u_flat = scipy.sparse.linalg.spsolve(self.A, self.f)

        u = u_flat.reshape([self.n_side, self.n_side])
        # Pad the zeros on each side... (due to the boundary conditions) and make the 
        # dyadic piecewise linear function object
        self.u.values = np.pad(u, ((1,1),(1,1)), 'constant')



"""
*****************************************************************************************
All the functions below are for building specific basis systems, reduced basis, sinusoid, 
coarse grid hat functions, etc...
*****************************************************************************************
"""

def make_pw_hat_basis(div):
    # Makes a complete hat basis for division div
    Vn = []
    # n is the number of internal grid points, i.e. we will have n different hat functionsdd
    # for our coarse-grid basis
    side_n = 2**div-1

    for k in range(side_n):
        for l in range(side_n):
            v = PWLinearSqDyadicH1(div=div)
            v.values[k+1, l+1] = 1.0
            Vn.append(v)
    
    b = PWBasis(Vn)

    h = 2 ** (-b.vecs[0].div)
    # We construct the Grammian here explicitly, otherwise it takes *forever*
    # as the grammian is often used in Reisz representer calculations
    grammian = np.zeros([side_n*side_n, side_n*side_n])
    diag = (4.0 + h*h/2.0) * np.ones(side_n*side_n)
    lr_diag = (h*h/12.0 - 1) * np.ones(side_n*side_n)

    # min_diag is below the diagonal, hence deals with element to the left in the FEM grid
    lr_diag[side_n-1::side_n] = 0 # These corresponds to edges on left or right extreme
    lr_diag = lr_diag[:-1]

    ud_diag = (h*h/12.0 - 1) * np.ones(side_n*side_n)
    ud_diag = ud_diag[side_n:]
    
    grammian = scipy.sparse.diags([diag, lr_diag, lr_diag, ud_diag, ud_diag], [0, -1, 1, -side_n, side_n]).tocsr()
    b.G = grammian
    
    return b

def make_pw_sin_basis(div, N=None, space='H1'):
    Vn = []

    if N is None:
        N = 2**div - 1

    # We want an ordering such that we get (1,1), (1,2), (2,1), (2,2), (2,3), (3,2), (3,1), (1,3), ...
    for n in range(1,N+1):
        for m in range(1,n+1):
            def f(x,y): return np.sin(n * math.pi * x) * np.sin(m * math.pi * y) * 2.0 / math.sqrt(1.0 + math.pi * math.pi * (m * m + n * n))
            v_i = PWLinearSqDyadicH1(func = f, div = div)
            Vn.append(v_i)
            
            # We do the mirrored map here
            if m < n:
                def f(x,y): return np.sin(m * math.pi * x) * np.sin(n * math.pi * y) * 2.0 / math.sqrt(1.0 + math.pi * math.pi * (m * m + n * n))

                v_i = PWLinearSqDyadicH1(func = f, div = div)
                Vn.append(v_i)

    return PWBasis(Vn, space=space)


def make_pw_reduced_basis(n, field_div, fem_div, point_gen=None, space='H1', a_bar=1.0, c=0.5, f=1.0):
    # Make a basis of m solutions to the FEM problem, from random generated fields

    side_n = 2**field_div
    
    if point_gen is None:
        point_gen = pg.MonteCarlo(d=side_n*side_n, n=n, lims=[-1, 1])
    elif point_gen.n != n:
        raise Exception('Need point dictionary with right number of points!')

    Vn = []
    fields = []

    for i in range(n):
        field = PWConstantSqDyadicL2(a_bar + c * point_gen.points[i,:].reshape([side_n, side_n]), div=field_div)
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        Vn.append(fem_solver.u)
        
    return PWBasis(Vn, space=space), fields

def make_pw_random_local_integration_basis(m, div, width=2, bounds=None, bound_prop=1.0, space='H1', return_map=False):

    M_m = []
    
    full_points =  list(product(range(2**div - (width-1)), range(2**div - (width-1))))

    if bounds is not None:
        bound_points = list(product(range(bounds[0,0], bounds[0,1] - (width-1)), range(bounds[1,0], bounds[1,1] - (width-1))))
        remain_points = [p for p in full_points if p not in bound_points]
        remain_locs = np.random.choice(range(len(remain_points)), round(m * (1.0 - bound_prop)), replace=False)
    else:
        bound_points = full_points 
        remain_points = [] 
        remain_locs = []
        
    bound_locs = np.random.choice(range(len(bound_points)), round(m * bound_prop), replace=False)
    
    points = [bound_points[bl] for bl in bound_locs] + [remain_points[rl] for rl in remain_locs]
    
    #np.random.choice(range(len(points)), m, replace=False)
    h = 2**(-div)

    local_meas_fun = PWConstantSqDyadicL2(div=div)
    
    stencil = h*h*3.0 * np.ones([width, width])
    stencil[0,:]=stencil[-1,:]=stencil[:,0]=stencil[:,-1]=h*h*3.0/2.0
    stencil[0,0]=stencil[-1,-1]=h*h/2.0
    stencil[0,-1]=stencil[-1,0]=h*h
    
    if space == 'H1':
        hat_b = make_pw_hat_basis(div=div)
        hat_b.make_grammian()
    for i in range(m):
        point = points[i]

        local_meas_fun.values[point[0]:point[0]+width,point[1]:point[1]+width] += 1.0

        meas = PWLinearSqDyadicH1(div=div)
        meas.values[point[0]:point[0]+width,point[1]:point[1]+width] = stencil

        if space == 'H1':
            # Then we have to make this an element of coarse H1,
            # which we do by creating a hat basis and solving
            if scipy.sparse.issparse(hat_b.G):
                v = scipy.sparse.linalg.spsolve(hat_b.G, meas.values[1:-1,1:-1].flatten())
            else:
                v = scipy.linalg.solve(hat_b.G, meas.values[1:-1,1:-1].flatten(), sym_pos=True)
            meas = hat_b.reconstruct(v)
            
        M_m.append(meas)
    
    W = PWBasis(M_m, space)
    if return_map:
        return W, local_meas_fun

    return W


def make_local_integration_basis(div, int_div, space='H1'):

    if div < int_div:
        raise Exception('Integration div must be less than or equal to field div')

    M_m = []
    side_m = 2**int_div
    h = 2**(-div)

    int_size = 2**(div - int_div)
    stencil = h*h*3.0 * np.ones([int_size+1, int_size+1])
    stencil[0,:]=stencil[-1,:]=stencil[:,0]=stencil[:,-1]=h*h*3.0/2.0
    stencil[0,0]=stencil[-1,-1]=h*h/2.0
    stencil[0,-1]=stencil[-1,0]=h*h
   
    if space == 'H1':
        hat_b = make_pw_hat_basis(div=div, space='H1')
        hat_b.make_grammian()

    for i in range(side_m):
        for j in range(side_m):

            meas = PWLinearSqDyadicH1(div=div)
            meas.values[i*int_size:(i+1)*int_size+1, j*int_size:(j+1)*int_size+1] = stencil

            if space == 'H1':
                # Then we have to make this an element of coarse H1,
                # which we do by creating a hat basis and solving
                if scipy.sparse.issparse(hat_b.G):
                    v = scipy.sparse.linalg.spsolve(hat_b.G, meas.values[1:-1,1:-1].flatten())
                else:
                    v = scipy.linalg.solve(hat_b.G, meas.values[1:-1,1:-1].flatten(), sym_pos=True)
                meas = hat_b.reconstruct(v)

            M_m.append(meas)
    
    W = PWBasis(M_m, space)
    return W
