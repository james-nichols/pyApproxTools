"""
vector.py

Author: James Ashton Nichols
Start date: June 2017

The abstraction of some linear algebra in Hilbert spaces for doing functional analysis 
computations, where the class "Basis" does some predictable operations on the 
class "Vector", which has to have a dot product defined as well as typical linear algebra.

In this submodule we define fundamental vector types, including our "FuncVector" which uses a 
dictionary approach to represent vectors as sums of exact functions, defined by different classes
of "Element". To implement this we use an extension of DefaultDict, called AlgebraDict, where 
for example adding two dicts results in the values of any corresponding keys being summed together,
and where any keys that aren't in exactly one vector get added to the final vector. In schematic
representation, if we had

v1 = [  1.4 * sin(2 * pi * x) +
        3.0 * sin(4 * pi * x) +
        1.0 * sin(10 * pi * x)
     ]
v2 = [  0.1 * sin(2 * pi * x) +
        1.0 * sin(10 * pi * x) +
     ]
then the result of 
v1 + v2 = [ 1.5 * sin(2 * pi * x) +
            3.0 * sin(4 * pi * x) +
            2.0 * sin(10 * pi * x) +
          ]

This operation is done by having a dictionary of "elements" which contain the simple dot-product
and evaluate routines, as well as 
"""

import math
import numpy as np
import scipy as sp
import collections # For defaultdict
import copy

import pdb

__all__ = ['AlgebraDict', 'Element', 'L2UIElement', 'L2UIHeaviside', 'L2UISin', 'L2UIAvg', 'H1UIElement', 'H1UIDelta', 'H1UIAvg', 'H1UISin', 'H1UIPoly', 'Vector', 'FuncVector']

class AlgebraDict(collections.defaultdict):
    """ A dictionary with algegraeic capability, used for exact function/vector representation """

    def keys_array(self):
        result = np.array(list(self.keys()))
        if result.dtype not in np.ScalarType:
            raise Exception('Error: params are not of consistent size')
        return result

    def values_array(self):
        return np.array(list(self.values()))

    def __add__(self, other):
        result = copy.deepcopy(self)
        for key, val in other.items():
            result[key] += val
        return result

    __radd__ = __add__

    def __iadd__(self, other):
        for key, val in other.items():
            self[key] += val
        return self
     
    def __sub__(self, other):
        result = copy.deepcopy(self)
        for key, val in other.items():
            result[key] -= val
        return result

    __rsub__ = __sub__

    def __isub__(self, other):
        for key, val in other.items():
            self[key] -= val
        return self

    def __neg__(self):
        result = copy.deepcopy(self)
        for k in result:
            result[k] = -result[k]
        return result
 
    def __pos__(self):
        result = copy.deepcopy(self)
        for k in result:
            result[k] = +result[k]
        return result

    def __mul__(self, other):
        """ other must be a scalar here """
        result = copy.deepcopy(self)
        for k in result:
            result[k] *= other
        return result

    __rmul__ = __mul__

    def __truediv__(self, other):
        """ other must be a scalar here """
        result = copy.deepcopy(self)
        for k in result:
            result[k] /= other
        return result

class Element(object):
    """ For vectors that are made up of "exact" functions, we allow them to be sums of
        Elements, typically some simple function like sin, delta or polynomial. This
        covers a surprising amount of functional analysis and algorithms. The main thing
        is that we have a hashing function by parameter type so that we can use the dictionary
        approach in ExactVector """

    def __init__(self):

        self.d = None 
        self.domain = None
        self.space = None

    def dot(self, right, left_params, right_params):
        """ These exact functions have mathematically defined dot products"""
        pass

    def evaluate(self, params, x):
        """ This returns an array of len(params) * len(x), i.e. x is the 2nd coord,
            which is an important distinction between Element and Vector:
            Vector will return an array of len(x), summing over all elements.
            Here we use evaluate for dot products, so must provide all evaluation """
        pass

    def _normaliser(self, params):
        pass

    def _delta_dot(self, left, left_params, right_params):
        """ Dotting with a delta function is always the same... """
        rc = right_params.values_array()

        x0 = left_params.keys_array()
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]
        
        return (ln * lc * self.evaluate(right_params, x0)).sum()

    def _avg_dot(self, right, left_params, right_params):
        # This is another one that the function should know about - how to do a
        # local average / integration
        pass

    def __hash__(self):
        return hash((type(self), self.d, self.domain))

    def __eq__(self, other):
        return type(self) is type(other) and self.d == other.d and self.domain == other.domain

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):
        return self.__class__.__name__

class H1UIElement(Element):
    """ The class of all elements that live in the Sobolev space H1 on the Unit Interval """
    
    def __init__(self):
        super().__init__()
        self.d = 1
        self.domain = (0,1)
        self.space = 'H1'

    def latex_str(self, params):
        return r'\mathrm{func}_{{{{0}}}}(x)'.format(params)

class L2UIElement(Element):
    """ The class of all elements that live in L2 on the Unit Interval """
    def __init__(self):
        super().__init__()
        self.d = 1
        self.domain = (0,1)
        self.space = 'L2'

    def latex_str(self, params):
        return r'\mathrm{{func}}_{{{{0}}}}(x)'.format(params)


class L2UIHeaviside(L2UIElement):
    
    def evaluate(self, params, x):
        m = params.keys_array()
        c = params.values_array()

        return c * (x[:,np.newaxis] <= m)

    def dot(self, right, left_params, right_params):
        return right._heav_dot(self, left_params, right_params)

    def _sin_dot(self, left, left_params, right_params):
        lc = left_params.values_array()[:,np.newaxis]
        lp = left_params.keys_array()[:,np.newaxis]
        rc = right_params.values_array()
        rp = right_params.keys_array()

        return (lc * rc * (1.0 - np.cos(np.pi * lp * rp))/(np.pi * lp)).sum()

    def _avg_dot(self, left, left_params, right_params):
        lc = left_params.values_array()[:,np.newaxis]
        a = left_params.keys_array()[:,0][:,np.newaxis]
        b = left_params.keys_array()[:,1][:,np.newaxis]
        rp = right_params.keys_array()
        rc = right_params.values_array()

        return (lc * rc * ((b - a) * (b <= rp) + (rp - a) * (rp > a) * (rp <= b))).sum()
        
    def _heav_dot(self, left, left_params, right_params):
        return self._self_dot(left_params, right_params)

    def _self_dot(self, left_params, right_params):
        lc = left_params.values_array()[:,np.newaxis]
        lp = left_params.keys_array()[:,np.newaxis]
        rc = right_params.values_array()
        rp = right_params.keys_array()

        return (lc * rc * ( (lp >= rp) * rp + (lp < rp) * lp)).sum()

class L2UISin(L2UIElement):

    def evaluate(self, params, x):
        m = params.keys_array()
        c = params.values_array()
        return c * np.sin(math.pi * np.outer(x, m))
    
    def dot(self, right, left_params, right_params):
        return right._sin_dot(self, left_params, right_params)

    def _heav_dot(self, left, left_params, right_params):
        lc = left_params.values_array()[:,np.newaxis]
        lp = left_params.keys_array()[:,np.newaxis]
        rc = right_params.values_array()
        rp = right_params.keys_array()

        return (lc * rc * (1.0 - np.cos(np.pi * lp * rp))/(np.pi * rp)).sum()
        
    def _sin_dot(self, left, left_params, right_params):
        return self._self_dot(left_params, right_params)

    def _self_dot(self, left_params, right_params):
        lc = left_params.values_array()[:,np.newaxis]
        lp = left_params.keys_array()[:,np.newaxis]
        rc = right_params.values_array()
        rp = right_params.keys_array()

        return (lc * rc * 0.5 * (lp == rp)).sum()
    
    def _avg_dot(self, left, left_params, right_params):
        lc = left_params.values_array()[:,np.newaxis]
        lp = left_params.keys_array()
        a = lp[:,0][:,np.newaxis]
        b = lp[:,1][:,np.newaxis]
        rc = right_params.values_array()
        rp = right_params.keys_array()
        
        return (lc * rc * (np.cos(np.pi * a * rp) - np.cos(np.pi * b * rp)) / (np.pi * rp)).sum()

class L2UIAvg(L2UIElement):

    def evaluate(self, params, x):

        a = params.keys_array()[:,0]
        b = params.keys_array()[:,1]
        c = params.values_array()

        if any(a > b):
            raise Exception('Some local-average intervals arprint(self.names_array(0))e in reverse, a > b')

        mid = (a <= x[:,np.newaxis]) & (x[:,np.newaxis] < b) #np.greatereq.outer(x, a) & np.less.outer(x, b)
        
        return c * mid

    def dot(self, right, left_params, right_params):
        return right._avg_dot(self, left_params, right_params)

    def _avg_dot(self, left, left_params, right_params):
        return self._self_dot(left_params, right_params)

    def _heav_dot(self, left, left_params, right_params):
        lc = left_params.values_array()[:,np.newaxis]
        lp = left_params.keys_array()[:,np.newaxis]
        rc = right_params.values_array()
        a = right_params.keys_array()[:,0]
        b = right_params.keys_array()[:,1]

        return (lc * rc * ((b - a) * (b <= lp) + (lp - a) * (lp > a) * (lp <= b))).sum()
        
    def _sin_dot(self, left, left_params, right_params):
        lc = left_params.values_array()[:,np.newaxis]
        lp = left_params.keys_array()[:,np.newaxis]
        rc = right_params.values_array()
        a = right_params.keys_array()[:,0]
        b = right_params.keys_array()[:,1]
        
        if any(a >= b): 
            raise Exception('Some local-average intervals are in reverse or null, a >= b')
        
        return (lc * rc * (np.cos(np.pi * a * lp) - np.cos(np.pi * b * lp)) / (np.pi * lp)).sum()

    def _self_dot(self, left_params, right_params):
        a = left_params.keys_array()[:,0][:,np.newaxis]
        b = left_params.keys_array()[:,1][:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]

        c = right_params.keys_array()[:,0]
        d = right_params.keys_array()[:,1]
        rc = right_params.values_array()

        if any(a >= b) or any(c >= d): 
            raise Exception('Some local-average intervals are in reverse or null, a >= b')
        
        inq_1 = a < c
        inq_2 = b < d
        inq_3 = a < d
        inq_4 = b < c

        # These selections are the only possible combinations of inq_1, 2, 3 and 4 
        # that don't clash with the requirements that a < b and c < d,
        # e.g. ~(a < c) & (b < c) => b < a...
        # Try the logic table if this confuses you...
        dot = (inq_1 & inq_2 & inq_3 & ~inq_4) * (b - c)
        dot += (inq_1 & ~inq_2 & inq_3 & ~inq_4) * (d - c)
        dot += (~inq_1 & inq_2 & inq_3 & ~inq_4) * (b - a)
        dot += (~inq_1 & ~inq_2 & inq_3 & ~inq_4) * (d - a)
        
        return (lc * rc * dot).sum()

class H1UISin(H1UIElement):

    def evaluate(self, params, x):
        m = params.keys_array()
        c = params.values_array()
        return c * np.sin(math.pi * np.outer(x, m)) * self._normaliser(params)
    
    def dot(self, right, left_params, right_params):
        return right._sin_dot(self, left_params, right_params)

    def _sin_dot(self, left, left_params, right_params):
        return self._self_dot(left_params, right_params)

    def _self_dot(self, left_params, right_params):
        lc = left_params.values_array()
        rc = right_params.values_array()
        lp = left_params.keys_array()
        rp = right_params.keys_array()

        return (lc[:,np.newaxis] * rc * np.equal.outer(lp, rp)).sum()

    def _avg_dot(self, left, left_params, right_params):
        a = left_params.keys_array()[:, 0][:,np.newaxis]
        b = left_params.keys_array()[:, 1][:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]

        m = right_params.keys_array()
        rc = right_params.values_array()
        rn = self._normaliser(right_params)
         
        result = ln * rn * lc * rc * (np.cos(math.pi * m * a) - np.cos(math.pi * m * b)) / (math.pi * m * (b - a))
        return result.sum()
    
    def _affine_dot(self, left, left_params, right_params):
        # affine params
        a = left_params.keys_array()[:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]
        
        # sin params
        m = right_params.keys_array()
        rc = right_params.values_array()
        rn = self._normaliser(right_params)

        #result = ln * lc * rn * rc * math.sqrt(2) * s * np.sin(m * math.pi * a)
        pn = (-1)**m
        result = ln * lc * rn * rc * 3 * ((a - 1) * pn - np.sin(m * math.pi * a) / (m * math.pi) ) / (m * math.pi)
        
        return result.sum()
    
    def _hat_dot(self, left, left_params, right_params):
        # hat params
        ln = left._normaliser(left_params)[:, np.newaxis]
        l, ml, m, mm, h, mh, lc = left._make_params(left_params, newaxis=True)

        # sin params
        n = right_params.keys_array()
        rc = right_params.values_array()
        rn = self._normaliser(right_params)
    
        pn = (-1)**n
        result = ln * lc * rn * rc * 3 * (ml * ((l - 1) * pn - np.sin(n * math.pi * l) / (n * math.pi))  \
                + mm * ((m - 1) * pn - np.sin(n * math.pi * m) / (n * math.pi) ) 
                + mh * ((h - 1) * pn - np.sin(n * math.pi * h) / (n * math.pi) )) / (n * math.pi)
        result[(l >= 1.0) | (m >= 1.0) | (h >= 1.0)] = 0.0
        return result.sum()

    def _normaliser(self, params):
        m = params.keys_array()
        return math.sqrt(2.0) / (math.pi * m)

    def latex_str(self, params):
        return r'\sin({{{0}}} \pi x)'.format(params)

class H1UIDelta(H1UIElement):

    def evaluate(self, params, x):
        # nb we allow both x0 and x to be np arrays
        # returns an array of size len(x) * len(x0)
        x0 = params.keys_array()
        c = params.values_array()

        # This is now a matrix of size len(x) * len(x0)
        choice = np.less.outer(x, x0) #np.array([x > x0ref for x0ref in x0])

        lower = self._normaliser(params) * np.outer(x, (1. - x0))
        upper = self._normaliser(params) * np.outer((1. - x), x0)
        
        if np.isscalar(choice):
            return lower * choice + upper * (not choice)

        return c * (lower * choice + upper * (~choice))
    
    def dot(self, right, left_params, right_params):
        return right._delta_dot(self, left_params, right_params)

    def _sin_dot(self, left, left_params, right_params):
        return left._delta_dot(self, right_params, left_params)
    def _avg_dot(self, left, left_params, right_params):
        return left._delta_dot(self, right_params, left_params)
    def _affine_dot(self, left, left_params, right_params):
        return left._delta_dot(self, right_params, left_params)
    def _hat_dot(self, left, left_params, right_params):
        return left._delta_dot(self, right_params, left_params)

    def _normaliser(self, params):
        p = params.keys_array()
        return 1. / np.sqrt((1. - p) * p)

    def latex_str(self, params):
        return r'\delta_{{{0}}} (x)'.format(params)

class H1UIAvg(H1UIElement):

    def evaluate(self, params, x):
        
        a = params.keys_array()[:,0]
        b = params.keys_array()[:,1]
        c = params.values_array()

        if any(a > b):
            raise Exception('Some local-average intervals are in reverse, a > b')

        low = x[:,np.newaxis] < a #np.less.outer(x, a)
        mid = (a <= x[:,np.newaxis]) & (x[:,np.newaxis] < b) #np.greatereq.outer(x, a) & np.less.outer(x, b)
        hi = b <= x[:,np.newaxis] #np.greatereq.outer(x, b)
        
        l = (1 - 0.5 * (a+b)) * x[:,np.newaxis]
        m = l - 0.5 * (a - x[:,np.newaxis]) * (a - x[:,np.newaxis]) / (b - a)
        h = 0.5 * (a + b) * (1.0 - x[:,np.newaxis])
        
        return c * self._normaliser(params) * (low * l + m * mid + h * hi)
        
    def dot(self, right, left_params, right_params):
        return right._avg_dot(self, left_params, right_params)
    
    def _avg_dot(self, right, left_params, right_params):
        return self._self_dot(left_params, right_params)

    def _self_dot(self, left_params, right_params):
        a = left_params.keys_array()[:,0][:,np.newaxis]
        b = left_params.keys_array()[:,1][:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = self._normaliser(left_params)[:,np.newaxis]

        c = right_params.keys_array()[:,0]
        d = right_params.keys_array()[:,1]
        rc = right_params.values_array()
        rn = self._normaliser(right_params)

        if any(a >= b) or any(c >= d): 
            raise Exception('Some local-average intervals are in reverse or null, a >= b')
        
        inq_1 = a < c
        inq_2 = b < d
        inq_3 = a < d
        inq_4 = b < c

        # These selections are the only possible combinations of inq_1, 2, 3 and 4 
        # that don't clash with the requirements that a < b and c < d,
        # e.g. ~(a < c) & (b < c) => b < a...
        # Try the logic table if this confuses you...
        dot =  (inq_1 & inq_2 & inq_3 & inq_4) * self._disj(a,b,c,d)
        dot += (inq_1 & inq_2 & inq_3 & ~inq_4) * self._intr(a,b,c,d)
        dot += (inq_1 & ~inq_2 & inq_3 & ~inq_4) * self._cont(a,b,c,d)
        dot += (~inq_1 & inq_2 & inq_3 & ~inq_4) * self._cont(c,d,a,b)
        dot += (~inq_1 & ~inq_2 & inq_3 & ~inq_4) * self._intr(c,d,a,b)
        dot += (~inq_1 & ~inq_2 & ~inq_3 & ~inq_4) * self._disj(c,d,a,b)
        
        return (lc * ln * rc * rn * dot).sum()
        
    # These internal functions represent the different cases for when the local avg is dotted against itself
    def _disj(self, a, b, c, d):
        return (1.0 - 0.5 * (c + d)) * 0.5 * (a + b)
    def _intr(self, a, b, c, d):
        return (1.0 - 0.5 * (c + d)) * 0.5 * (a + b) - (b - c)**3 / (6.0 * (b - a) * (d - c))
    def _cont(self, a, b, c, d):
        return (1.0/(b-a)) * ((1 - 0.5 * (c + d)) * 0.5 * (d*d - a*a) - (d - c)*(d - c) / 6.0 \
                - 0.25 * (c + d) * ((1-b)*(1-b) - (1-d)*(1-d)))

    def _sin_dot(self, left, left_params, right_params):
        # sin params
        m = left_params.keys_array()[:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]

        # avg params
        a = right_params.keys_array()[:, 0]
        b = right_params.keys_array()[:, 1]
        rc = right_params.values_array()
        rn = self._normaliser(right_params)
         
        result = ln * rn * lc * rc * (np.cos(math.pi * m * a) - np.cos(math.pi * m * b)) / (math.pi * m * (b - a))
        return result.sum()

    def _affine_dot(self, left, left_params, right_params):
        # affine params
        d = left_params.keys_array()[:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]

        # avg params
        a = right_params.keys_array()[:, 0]
        b = right_params.keys_array()[:, 1]
        rc = right_params.values_array()
        rn = self._normaliser(right_params)

        result = ln * lc * rn * rc * self._inner_aff_avg(a, b, d)
        return np.nan_to_num(result).sum()

    def _hat_dot(self, left, left_params, right_params):
        # affine params
        ln = left._normaliser(left_params)[:,np.newaxis]
        l, ml, m, mm, h, mh, lc = left._make_params(left_params, newaxis=True)
        
        # avg params
        a = right_params.keys_array()[:, 0]
        b = right_params.keys_array()[:, 1]
        rc = right_params.values_array()
        rn = self._normaliser(right_params)
        
        result = ln * lc * rn * rc * (ml * self._inner_aff_avg(a, b, l) \
                                     + mm * self._inner_aff_avg(a, b, m) \
                                     + mh * self._inner_aff_avg(a, b, h))
        return np.nan_to_num(result).sum()
    
    def _inner_aff_avg(self, a, b, d):
        result = 0.5 * (0.5 * (b*b - a*a) * (1-d)**3 - (d < b) * 0.25 * (b-d)**4 + (d < a) * 0.25 * (a-d)**4) / (b - a)
        result[(d >= 1.0)] = 0.0
        return result

    def _normaliser(self, params):
        p = params.keys_array()
        return 1.0 / np.sqrt(p[:,0] + (p[:,1] - p[:,0])/3.0 - 0.25 * (p[:,0] + p[:,1]) * (p[:,0] + p[:,1]))

    def _nt(self, a, b):
        return 1.0 / np.sqrt(a + (b - a)/3.0 - 0.25 * (a + b) * (a + b))

    def latex_str(self, params):
        return r'\mathbb{1}' + '_{{{0}}}(x)'.format(params)

class H1UIAffine(H1UIElement):

    def evaluate(self, params, x):
        
        a = params.keys_array()
        c = params.values_array()
        
        hi = x[:,np.newaxis] > a #np.less.outer(x, a)
        
        return c * self._normaliser(params) * 0.5 * (x[:,np.newaxis] * (1.0 - a)**3 - hi * (x[:,np.newaxis] - a)**3)
        
    def dot(self, right, left_params, right_params):
        return right._affine_dot(self, left_params, right_params)
    
    def _affine_dot(self, left, left_params, right_params):
        return self._self_dot(left_params, right_params)
    
    def _self_dot(self, left_params, right_params):
        # The algorithm is ordered, so we must order it
        a = left_params.keys_array()[:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = self._normaliser(left_params)[:,np.newaxis]

        b = right_params.keys_array()
        rc = right_params.values_array()
        rn = self._normaliser(right_params)

        valid = (a >= 0) & (a < 1) & (b >= 0) & (b < 1)
        order = (a <= b)

        result = lc * ln * rc * rn * (order * self._aff_ordered_dot(a, b) + ~order * self._aff_ordered_dot(b, a))
        return np.nan_to_num(result).sum()

    def _aff_ordered_dot(self, a, b):
        # This routine assumes a <= b. Also assumes that, if a and b are vectors, that a is a row vector, i.e.
        # that we have done the a[:, np.newaxis] step
        d = 0.25 * (9*(0.20 * (1-b**5) - 0.5 * (a + b) * (1 - b**4) + (a*a + 4*a*b + b*b) * (1-b**3) / 3 \
                    - a * b * (a + b) * (1 - b*b) + a * a * b * b * (1 - b)) - (1-a)**3 * (1-b)**3)
        return d

    def _sin_dot(self, left, left_params, right_params):
        # sin params
        m = left_params.keys_array()[:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]
        
        # affine params
        a = right_params.keys_array()
        rc = right_params.values_array()
        rn = self._normaliser(right_params)

        #result = ln * lc * rn * rc * math.sqrt(2) * s * np.sin(m * math.pi * a)
        
        pn = (-1)**m
        result = ln * lc * rn * rc * 3 * ((a - 1) * pn - np.sin(m * math.pi * a) / (m * math.pi) ) / (m * math.pi)
        return np.nan_to_num(result).sum()

    def _avg_dot(self, left, left_params, right_params):
        # avg params
        a = left_params.keys_array()[:, 0][:,np.newaxis]
        b = left_params.keys_array()[:, 1][:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]
        
        # affine params
        d = right_params.keys_array()
        rc = right_params.values_array()
        rn = self._normaliser(right_params)

        Idb = d < b
        Ida = d < a
        
        #result = ln * lc * rn * rc * 0.5 * s * ((1-d)**3 * (b - a) - Idb * (b-d)**3 + Ida * (a-d)**3)
        result = ln * lc * rn * rc * 0.5 * (0.5 * (b*b - a*a) * (1-d)**3 - Idb * 0.25 * (b-d)**4 + Ida * 0.25 * (a-d)**4) / (b - a)
        return np.nan_to_num(result).sum()

    def _hat_dot(self, left, left_params, right_params):
        # hat params
        ln = left._normaliser(left_params)[:,np.newaxis]
        l, ml, m, mm, h, mh, lc = left._make_params(left_params, newaxis=True)

        # affine params
        a = left_params.keys_array()[:,np.newaxis]
        rc = left_params.values_array()[:,np.newaxis]
        rn = left._normaliser(left_params)[:,np.newaxis]
 
        # Requires checking the order....
        d = ml * (self._aff_ordered_dot(l, a) * (l <= a) + self.aff_ordered_dot(a, l) * (a < l))
        d += mm * (self._aff_ordered_dot(m, a) * (m <= a) + self.aff_ordered_dot(a, m) * (a < m))
        d += mh * (self._aff_ordered_dot(h, a) * (h <= a) + self.aff_ordered_dot(a, h) * (a < h))
        
        return (ln * lc * rn * rc * d).sum()

    def _normaliser(self, params):
        p = params.keys_array()
        result = 1.0 / np.sqrt(self._aff_ordered_dot(p, p))
        result[p >= 1.0] = 0.0
        return result

    def latex_str(self, params):
        return r'U(x - {{{0}}}) (x - {{{0}}})'.format(params[0])

class H1UIHat(H1UIElement):
    """ Hat function between a and b, normalised in L2, then in H1 """

    def __init__(self):
        super().__init__()
        self.f = H1UIAffine()
 
    def _old_evaluate(self, params, x):
        d1, d2, d3 = self._make_dicts(params)
        return self._normaliser(params) * (self.f.evaluate(d1, x) + self.f.evaluate(d2, x) + self.f.evaluate(d3, x))
 
    def evaluate(self, params, x):
        l, ml, m, mm, h, mh, c = self._make_params(params)
        return c * self._normaliser(params) * (ml*self._inner_evaluate(l, x) + mm*self._inner_evaluate(m, x) + mh*self._inner_evaluate(h, x))

    def _inner_evaluate(self, a, x):
        hi = x[:,np.newaxis] > a #np.less.outer(x, a)
        return 0.5 * (x[:,np.newaxis] * (1.0 - a)**3 - hi * (x[:,np.newaxis] - a)**3)

    def _aff_ordered_dot(self, a, b):
        # This routine assumes a <= b. Also assumes that, if a and b are vectors, that a is a row vector, i.e.
        # that we have done the a[:, np.newaxis] step
        return 0.25 * (9*(0.20 * (1-b**5) - 0.5 * (a + b) * (1 - b**4) + (a*a + 4*a*b + b*b) * (1-b**3) / 3 \
                    - a * b * (a + b) * (1 - b*b) + a * a * b * b * (1 - b)) - (1-a)**3 * (1-b)**3)

    def dot(self, right, left_params, right_params):
        return right._hat_dot(self, left_params, right_params)
    
    def _hat_dot(self, left, left_params, right_params):
        # affine params
        ln = self._normaliser(left_params)[:, np.newaxis]
        l1, ml1, m1, mm1, h1, mh1, lc = self._make_params(left_params, newaxis=True)

        # affine params
        rn = self._normaliser(right_params)
        l2, ml2, m2, mm2, h2, mh2, rc = self._make_params(right_params)
        
        # Requires checking the order....
        d = ml1 * ml2 * (self._aff_ordered_dot(l1, l2) * (l1 <= l2) + self._aff_ordered_dot(l2, l1) * (l2 < l1))
        d += ml1 * mm2 * (self._aff_ordered_dot(l1, m2) * (l1 <= m2) + self._aff_ordered_dot(m2, l1) * (m2 < l1)) 
        d += ml1 * mh2 * (self._aff_ordered_dot(l1, h2) * (l1 <= h2) + self._aff_ordered_dot(h2, l1) * (h2 < l1))
        d += mm1 * ml2 * (self._aff_ordered_dot(m1, l2) * (m1 <= l2) + self._aff_ordered_dot(l2, m1) * (l2 < m1))
        d += mm1 * mm2 * (self._aff_ordered_dot(m1, m2) * (m1 <= m2) + self._aff_ordered_dot(m2, m1) * (m2 < m1))
        d += mm1 * mh2 * (self._aff_ordered_dot(m1, h2) * (m1 <= h2) + self._aff_ordered_dot(h2, m1) * (h2 < m1))
        d += mh1 * ml2 * (self._aff_ordered_dot(h1, l2) * (h1 <= l2) + self._aff_ordered_dot(l2, h1) * (l2 < h1))
        d += mh1 * mm2 * (self._aff_ordered_dot(h1, m2) * (h1 <= m2) + self._aff_ordered_dot(m2, h1) * (m2 < h1))
        d += mh1 * mh2 * (self._aff_ordered_dot(h1, h2) * (h1 <= h2) + self._aff_ordered_dot(h2, h1) * (h2 < h1))
        
        return (ln * lc * rn * rc * d).sum()

    def _affine_dot(self, left, left_params, right_params):
        # affine params
        a = left_params.keys_array()[:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]

        # hat params
        rn = self._normaliser(right_params)
        l, ml, m, mm, h, mh, rc = self._make_params(right_params)
        
        # Requires checking the order....
        d = ml * (self._aff_ordered_dot(l, a) * (l <= a) + self._aff_ordered_dot(a, l) * (a < l))
        d += mm * (self._aff_ordered_dot(m, a) * (m <= a) + self._aff_ordered_dot(a, m) * (a < m))
        d += mh * (self._aff_ordered_dot(h, a) * (h <= a) + self._aff_ordered_dot(a, h) * (a < h))
        
        return (ln * lc * rn * rc * d).sum()

    def _sin_dot(self, left, left_params, right_params):
        # sin params
        n = left_params.keys_array()[:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]
        
        # affine params
        rn = self._normaliser(right_params)
        l, ml, m, mm, h, mh, rc = self._make_params(right_params)
        
        pn = (-1)**n
        result = ln * lc * rn * rc * 3 * (ml * ((l - 1) * pn - np.sin(n * math.pi * l) / (n * math.pi))  \
                + mm * ((m - 1) * pn - np.sin(n * math.pi * m) / (n * math.pi) ) \
                + mh * ((h - 1) * pn - np.sin(n * math.pi * h) / (n * math.pi) )) / (n * math.pi)
        result[(l >= 1.0) | (m >= 1.0) | (h >= 1.0)] = 0.0
        return resultnext(iter(self.vecs[-1].elements.values()))

    def _avg_dot(self, left, left_params, right_params):
        # avg params
        a = left_params.keys_array()[:, 0][:,np.newaxis]
        b = left_params.keys_array()[:, 1][:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = left._normaliser(left_params)[:,np.newaxis]
        
        # affine params
        rn = self._normaliser(right_params)
        l, ml, m, mm, h, mh, rc = self._make_params(right_params)

        #result = ln * lc * rn * rc * 0.5 * s * ((1-d)**3 * (b - a) - Idb * (b-d)**3 + Ida * (a-d)**3)
        result = ln * lc * rn * rc * (ml * self._inner_aff_avg(a, b, l) \
                                     + mm * self._inner_aff_avg(a, b, m) \
                                     + mh * self._inner_aff_avg(a, b, h))
        return np.nan_to_num(result).sum()
    
    def _inner_aff_avg(self, a, b, d):
        return 0.5 * (0.5 * (b*b - a*a) * (1-d)**3 - (d < b) * 0.25 * (b-d)**4 + (d < a) * 0.25 * (a-d)**4) / (b - a)

    def _make_dicts(self, params):
        l, ml, m, mm, h, mh, c = self._make_params(params)

        # Argh have to make new dicts
        d1 = AlgebraDict(float, zip(l, c * ml))
        d2 = AlgebraDict(float, zip(m, c * mm))
        d3 = AlgebraDict(float, zip(h, c * mh))

        return d1, d2, d3

    def _make_params(self, params, newaxis=False):
        a = params.keys_array()[:,0]
        b = params.keys_array()[:,1]
        c = params.values_array()

        m = 4 / ((b - a) * (b - a))

        if newaxis:
            return a[:,np.newaxis], m[:,np.newaxis], (0.5*(a+b))[:,np.newaxis], (-2*m)[:,np.newaxis], \
                    b[:,np.newaxis], m[:,np.newaxis], c[:,np.newaxis]
        return a, m, 0.5*(a+b), -2*m, b, m, c

    def _normaliser(self, params):
        l, ml, m, mm, h, mh, c = self._make_params(params)

        # p is assumed to be an array of size n*2
        n2 =  1.0 / np.sqrt(ml * ml * self._aff_ordered_dot(l,l) + mm * mm * self._aff_ordered_dot(m, m) + mh * mh * self._aff_ordered_dot(h, h) \
                + ml * mm * self._aff_ordered_dot(l, m) + ml * mh * self._aff_ordered_dot(l, h) + mm * mh * self._aff_ordered_dot(m, h))
        return n2

    def latex_str(self, params):
        return r'\mathrm{Hat}_{' + '{0},{1}'.format(params[0], params[1]) + '}(x)'

class H1UIPoly(H1UIElement):
    pass

class Vector(object):
    
    # Ok new paradigm - use numpy to be a bit faster...

    def __init__(self):
        pass

    def dot(self, other):
        pass
    
    def norm(self):
        return math.sqrt(self.dot(self))

    def evaluate(self, x):
        pass

# Worth considering: cleaner methods of instantiation using string specifications
# e.g. using Python in-build parser or compile()

class FuncVector(Vector):

    def __init__(self, **kwargs):
        # TODO: Consider giving AlgebraDict the notion of dot product and then allow
        # this notion of a function vector to be a recursive thing... Whoa!
        self.elements = kwargs.get('elements') or AlgebraDict(lambda: AlgebraDict(float))

        # If all of 'funcs', 'params' and 'coeffs' are provided, we then add them to the elements dictionary
        if kwargs.get('funcs') and kwargs.get('params') and kwargs.get('coeffs'):
            funcs = kwargs.get('funcs') 
            params = kwargs.get('params') 
            coeffs = kwargs.get('coeffs')

            if not len(funcs) == len(params) == len(coeffs):
                raise Exception('Error - number of funcs not same as number of parameters and coefficients')

            for func, param, coeff in zip(funcs, params, coeffs): 
                for p, c in zip(param, coeff):
                    if isinstance(p, (list, tuple, np.ndarray)):
                        self.elements[eval(func)()][p] += float(c)
                    else:
                        self.elements[eval(func)()][float(p)] += float(c)
    
    def __str__(self):
        string = 'FuncVector: {'
        for el in self.elements:
            string += str(el) + ': {'
            for i, p in enumerate(self.elements[el]):
                if i==0:
                    string += str(p) + ': ' + str(self.elements[el][p])
                else: 
                    string += ', ' + str(p) + ': ' + str(self.elements[el][p])
            string += '}'
        string += '}'
        return string
    
    def latex_str(self):
        string = r'$'
        first = True
        for el in self.elements:
            for i, p in enumerate(self.elements[el]):
                if first:
                    first = False
                else:
                    string += ' + '

                if self.elements[el][p] != 1.0:
                    string += str(self.elements[el][p]) 
                
                string += el.latex_str(p)
        string += '$'
        return string

    def names_array(self):
        return [str(el) for el in self.elements]
    def params_array(self, index):
        return self.elements.values_array()[index].keys_array()
    def coeffs_array(self, inded):
        return self.elements.values_array()[index].values_array()
    
    def dot(self, other):
        dot = 0.0
        for l in self.elements:
            for r in other.elements:
                dot += l.dot(r, self.elements[l], other.elements[r])
        return dot

    def evaluate(self, x):
        ev = 0.0
        for el in self.elements:
            ev += el.evaluate(self.elements[el], x).sum(axis=-1)
        return ev

    def __add__(self, other):
        result = copy.deepcopy(self)
        result.elements += other.elements
        return result

    __radd__ = __add__

    def __iadd__(self, other):
        self.elements += other.elements
        return self 
     
    def __sub__(self, other):
        result = copy.deepcopy(self)
        result.elements -= other.elements
        return result

    def __rsub__(self, other):
        return -self.__sub__(self, other)

    def __isub__(self, other):
        self.elements -= other.elements
        return self 

    def __neg__(self):
        result = copy.deepcopy(self)
        result.elements = -result.elements
        return result
 
    def __pos__(self):
        result = copy.deepcopy(self)
        result.elements = +result.elements
        return result

    def __mul__(self, other):
        result = copy.deepcopy(self)
        result.elements *= other
        return result

    __rmul__ = __mul__

    def __truediv__(self, other):
        result = copy.deepcopy(self)
        result.elements /= other
        return result


