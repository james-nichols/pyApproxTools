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

import warnings

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
            if func is not None:
                if values is not None:
                    raise Exception('{0}: Specify either a function or the values, not both'.format(self.__class__.__name__))
                x, y = np.meshgrid(self.x_grid, self.y_grid)
                self.values = func(x, y)
            ## TODO: put this logic in the values setter routine
            elif values is not None:
                self.values = np.copy(values)
            else:
                self.values = np.zeros([self.side_len, self.side_len])
        else:
            if values is not None:
                self.values = np.copy(values)
                
            elif func is not None:
                raise Exception('{0}: Error - need grid size when specifying function'.format(self.__class__.__name__))
    
    @property
    def side_len(self):
        return self._side_len(self.div)
    def _side_len(self, d):
        return 2**d

    @property
    def x_grid(self):
        return self._x_grid(self.side_len)
    def _x_grid(self, sl):
        return np.linspace(self.domain[0][0], self.domain[0][1], sl, endpoint=True)
    @property
    def y_grid(self):
        return self._y_grid(self.side_len)
    def _y_grid(self, sl):
        return np.linspace(self.domain[1][0], self.domain[1][1], sl, endpoint=True)

    @property
    def values(self):
        return self._values
    @values.setter
    def values(self, vals):
        self._set_values(vals)

    def _set_values(self, vals):
        if self.div is None:        
            # If we don't yet have a div, then we extract it
            self.div = 0
            while self.side_len < vals.shape[0]:
                self.div += 1

        if (vals.shape[0] != vals.shape[1] or vals.shape[0] != self.side_len):
            raise Exception("{0}: Error - values of shape {1}x{2} not square or don't match dimensions {3}x{3} (div {4})" .format(self.__class__.__name__, \
                            vals.shape[0], vals.shape[1], self.side_len, self.div))
        self._values = vals

    def interpolate(self, interp_div):
        """ Simple interpolation routine to make this function on a finer division dyadic grid """
        pass 

    # Here we overload the + += - -= * and / operators
    def __add__(self, other):
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
    
    def __rsub__(self, other):
        if isinstance(other, type(self)):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return type(self)(v.values - u.values, d)
        else:
            return type(self)(other - self.values, self.div)

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

    def __neg__(self):
        result = copy.deepcopy(self)
        result.values = -result.values
        return result
 
    def __pos__(self):
        result = copy.deepcopy(self)
        result.values = +result.values
        return result


class PWLinearSqDyadicH1(PWSqDyadic):
    """ Describes a piecewise linear function on a dyadic P1 tringulation of the unit square.
        Includes routines to calculate L2 and H1 dot products, and interpolate between different dyadic levels
        """

    def __init__(self, values = None, div = None, func = None):

        super().__init__(values, div, func)
        self.space = 'H1'

    def _side_len(self, d):
        return 2 ** d + 1

    def _set_values(self, vals):
        super()._set_values(vals)

        if not np.allclose(self.values[:,0], 0) or not np.allclose(self.values[:,-1], 0) or not np.allclose(self.values[0,:], 0) or not np.allclose(self.values[-1,:], 0):
            warnings.warn("{0}: attempted to set some boundary values as non-zero, were forced to zero".format(self.__class__.__name__))
        self.values[:,0] = self.values[:,-1] = self.values[0,:] = self.values[-1,:] = 0

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
        
        return 0.5 * dot # + self.L2_inner(u,v,h)

    def L2_inner_new_proposed(self, u, v, h):
        # u and v are on the same grid / triangulation, so now we do the simple L2
        # inner product (hah... simple??)

        # the point adjacency matrix
        p = 6 * (1.0/18.0) * np.ones(u.shape)
        p[:,0] = p[0,:] = p[:,-1] = p[-1,:] = 3
        p[0,0] = p[-1,-1] = 1
        p[0,-1] = p[-1, 0] = 2 
        dot = (u * v * p).sum()
        
        # Now add all the vertical edges
        p = 2 * (2.0/45.0) * np.ones([u.shape[0]-1, u.shape[1]])
        p[0,:] = p[-1,:] = 1
        dot = dot + ((u[1:,:] * v[:-1,:] + u[:-1,:] * v[1:,:]) * p * 0.5).sum()

        # Now add all the horizontal edges
        p = 2 * (2.0/45.0) * np.ones([u.shape[0], u.shape[1]-1])
        p[:,0] = p[:,-1] = 1
        dot = dot + ((u[:,1:] * v[:,:-1] + u[:,:-1] * v[:,1:]) * p * 0.5).sum()

        # Finally all the diagonals (note every diagonal is adjacent to two triangles,
        # so don't need p)
        dot = dot + 2 * (1.0/72.0) * (u[:-1,1:] * v[1:,:-1] + u[1:,:-1] * v[:-1,1:] ).sum()
        #dot = dot + (u[:-1,1:] * v[:-1,1:] + u[1:,:-1] * v[1:,:-1] ).sum()
        
        return h * h * dot

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
            return type(self)(interp_func(self._x_grid(self._side_len(interp_div)), self._y_grid(self._side_len(interp_div))), interp_div)

    def plot(self, ax, title=None, div_frame=4, alpha=0.5, cmap=cm.jet, show_axes_labels=True):

        xs, ys = np.meshgrid(self.x_grid, self.y_grid)

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

    def _side_len(self, d):
        return 2**d

    def _x_grid(self, sl):
        return np.linspace(self.domain[0][0], self.domain[0][1], sl, endpoint=False) + 0.5 / sl
    def _y_grid(self, sl):
        return np.linspace(self.domain[1][0], self.domain[1][1], sl, endpoint=False) + 0.5 / sl

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
            raise Exception('{0}: Interpolate div must be greater than or equal to field div'.format(self.__class__.__name__))
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

