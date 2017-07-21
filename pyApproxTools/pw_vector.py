"""
pw_vector.py

Author: James Ashton Nichols
Start date: June 2017

Code to deal with piece-wise linear functions on triangulations - allows for treatement of FEM solutions etc...
"""

import math
import numpy as np
import scipy.interpolate
import scipy.linalg
from scipy import sparse
import copy

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import axes3d, Axes3D

from pyApproxTools.vector import *
from pyApproxTools.basis import *

__all__ = ['PWSqDyadic', 'PWLinearSqDyadicH1', 'PWConstantSqDyadicL2']

# TODO: Have intermediate PW linear vector definitions on arbitrary spaces...

# TODO: Make dyadic PWLinear a form of "element", that way can combine w/ exact function representations...

class PWSqDyadic(Vector):

    def __init__(self, values=None, div=None, func=None):
        
        # Do we really put these properties here? Hmm.
        self.d = 2
        self.domain = ((0,1),(0,1))
        self.space = None
        self.div = div

        if div is not None:
            self.div = div
            self.x_grid = np.linspace(self.domain[0][0], self.domain[0][1], self.side_len, endpoint=True)
            self.y_grid = np.linspace(self.domain[1][0], self.domain[1][1], self.side_len, endpoint=True)
            
            if func is not None:
                if values is not None:
                    raise Exception('DyadicPWLinear: Specify either a function or the values, not both')
                x, y = np.meshgrid(self.x_grid, self.y_grid)
                self.values = func(x, y)
            elif values is not None:
                if (values.shape[0] != values.shape[1] or values.shape[0] != self.side_len):
                    raise Exception("DyadicPWLinear: Error - values must be on a dyadic square of size {0}".format(self.side_len))
                self.values = values

            else:
                self.values = np.zeros([self.side_len, self.side_len])
        else:
            if values is not None:
                self.values = values
                self.div = int(math.log(values.shape[0] - 1, 2))
                if (values.shape[0] != values.shape[1] or values.shape[0] != self.side_len):
                    raise Exception("DyadicPWLinear: Error - values must be on a dyadic square, shape of {0} closest to div {1}".format(self.side_len, self.div))
                self.x_grid = np.linspace(0.0, 1.0, self.side_len, endpoint=True)
                self.y_grid = np.linspace(0.0, 1.0, self.side_len, endpoint=True)
            elif func is not None:
                raise Exception('DyadicPWLinear: Error - need grid size when specifying function')
    
    @property
    def side_len(self):
        return 0

    def interpolate(self, interp_div):
        """ Simple interpolation routine to make this function on a finer division dyadic grid """
        pass 


    # Here we overload the + += - -= * and / operators
    def __add__(self,other):
        if isinstance(other, type(self)):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return type(self)(u.values + v.values, d)
        else:
            return type(self)(self.values + other, self.div)

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            self.div = d
            self.values = u.values + v.values
        else:
            self.values = self.values + other
        return self
        
    def __sub__(self, other):
        if isinstance(other, type(self)):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return type(self)(u.values - v.values, d)
        else:
            return type(self)(self.values - other, self.div)
    __rsub__ = __sub__

    def __isub__(self, other):
        if isinstance(other, type(self)):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            self.div = d
            self.values = u.values - v.values
        else:
            self.values = self.values - other
        return self

    def __mul__(self, other):
        if isinstance(other, type(self)):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return type(self)(u.values * v.values, d)
        else:
            return type(self)(self.values * other, self.div)
    __rmul__ = __mul__

    def __pow__(self,power):
        return type(self)(self.values**power, self.div)

    def __truediv__(self, other):
        if isinstance(other, type(self)):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return type(self)(u.values / v.values, d)
        else:
            return type(self)(self.values / other, self.div)

class PWLinearSqDyadicH1(PWSqDyadic):
    """ Describes a piecewise linear function on a dyadic P1 tringulation of the unit square.
        Includes routines to calculate L2 and H1 dot products, and interpolate between different dyadic levels
        """

    def __init__(self, values = None, div = None, func = None):

        super().__init__(values, div, func)
        self.space = 'H1'

    @property
    def side_len(self):
        return 2 ** self.div + 1

    def dot(self, other):
        if isinstance(other, type(self)):
            if other.space == self.space:
                if self.div == other.div:
                    return self.H1_dot(other)
                else:
                    d = max(self.div,other.div)
                    u = self.interpolate(d)
                    v = other.interpolate(d)
                    return u.H1_dot(v)
        else:
            # TODO: make this a warning rather than exception
            raise Exception('Dot product can only be between compatible dyadic functions')
    
    def H1_dot(self, other):
        """ Compute the H1_0 dot product with another DyadicPWLinear function
            automatically interpolates the coarser function """
        
        d = max(self.div,other.div)
        u = self.interpolate(d).values
        v = other.interpolate(d).values

        h = 2.0**(-d)
        n_side = 2**d

        # This is du/dy
        p = 2 * np.ones([n_side, n_side+1])
        p[:,0] = p[:,-1] = 1
        dot = (p * (u[:-1,:] - u[1:,:]) * (v[:-1,:] - v[1:,:])).sum()
        # And this is du/dx
        p = 2 * np.ones([n_side+1, n_side])
        p[0,:] = p[-1,:] = 1
        dot = dot + (p * (u[:,1:] - u[:,:-1]) * (v[:,1:] - v[:,:-1])).sum()
        
        return 0.5 * dot + self.L2_inner(u,v,h)

    def L2_inner(self, u, v, h):
        # u and v are on the same grid / triangulation, so now we do the simple L2
        # inner product (hah... simple??)

        # the point adjacency matrix
        p = 6 * np.ones(u.shape)
        p[:,0] = p[0,:] = p[:,-1] = p[-1,:] = 3
        p[0,0] = p[-1,-1] = 1
        p[0,-1] = p[-1, 0] = 2 
        dot = (u * v * p).sum()
        
        # Now add all the vertical edges
        p = 2 * np.ones([u.shape[0]-1, u.shape[1]])
        p[0,:] = p[-1,:] = 1
        dot = dot + ((u[1:,:] * v[:-1,:] + u[:-1,:] * v[1:,:]) * p * 0.5).sum()

        # Now add all the horizontal edges
        p = 2 * np.ones([u.shape[0], u.shape[1]-1])
        p[:,0] = p[:,-1] = 1
        dot = dot + ((u[:,1:] * v[:,:-1] + u[:,:-1] * v[:,1:]) * p * 0.5).sum()

        # Finally all the diagonals (note every diagonal is adjacent to two triangles,
        # so don't need p)
        dot = dot + (u[:-1,1:] * v[1:,:-1] + u[1:,:-1] * v[:-1,1:] ).sum()
        
        return h * h * dot / 12

    def interpolate(self, interp_div):
        """ Simple interpolation routine to make this function on a finer division dyadic grid """
        if interp_div < self.div:
            raise Exception("Interpolation division smaller than field division! Need to integrate")
        elif interp_div == self.div:
            return self
        else:
            interp_func = scipy.interpolate.interp2d(self.x_grid, self.y_grid, self.values, kind='linear')
            x = y = np.linspace(0.0, 1.0, 2**interp_div + 1, endpoint=True)
            return DyadicPWLinear(interp_func(x, y), interp_div)

    def plot(self, ax, title=None, div_frame=4, alpha=0.5, cmap=cm.jet, show_axes_labels=True):

        x = np.linspace(0.0, 1.0, self.values.shape[0], endpoint = True)
        y = np.linspace(0.0, 1.0, self.values.shape[1], endpoint = True)
        xs, ys = np.meshgrid(x, y)

        if self.div > div_frame:
            wframe = ax.plot_surface(xs, ys, self.values, cstride=2**(self.div - div_frame), rstride=2**(self.div-div_frame), 
                                     cmap=cmap, alpha=alpha)
        else:
            wframe = ax.plot_surface(xs, ys, self.values, cstride=1, rstride=1, cmap=cmap, alpha=alpha)

        ax.set_facecolor('white')
        if show_axes_labels:
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
        if title is not None:
            ax.set_title(title)

class PWConstantSqDyadicL2(PWSqDyadic):
    """ Describes a piecewise constant function on a dyadic P1 tringulation of the unit cube. """
    
    def __init__(self, values = None, div = None, func = None):

        super().__init__(values, div, func)
        self.space = 'L2'

    @property
    def side_len(self):
        return 2 ** self.div

    def dot(self, other):
        if isinstance(other, type(self)):
            return self.L2_dot(other)
        else:
            raise Exception('Dot product can only be between compatible dyadic functions')

    def L2_dot(self, other):

        d = max(self.div, other.div)
        u = self.interpolate(d)
        v = other.interpolate(d)

        return (u.values * v.values).sum() * 2**(-2 * d)

    def interpolate(self, div):
        """ Simple interpolation routine to make this function on a finer division dyadic grid """
        if div < self.div:
            raise Exception('DyadicPWConstant: Interpolate div must be greater than or equal to field div')
        elif div == self.div:
            return self
        else:
            return type(self)(values=self.values.repeat(2**(div-self.div), axis=0).repeat(2**(div-self.div), axis=1),
                                    div=div)

    def plot(self, ax, title=None, alpha=0.5, cmap=cm.jet, show_axes_labels=True):

        # We do some tricks here (i.e. using np.repeat) to plot the piecewise constant nature of the random field...
        x = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint = True).repeat(2)[1:-1]
        xs, ys = np.meshgrid(x, x)
        wframe = ax.plot_surface(xs, ys, self.values.repeat(2, axis=0).repeat(2, axis=1), cstride=1, rstride=1,
                                 cmap=cmap, alpha=alpha)
        
        ax.set_facecolor('white')
        if show_axes_labels:
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
        if title is not None:
            ax.set_title(title)

