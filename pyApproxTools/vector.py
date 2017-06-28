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
        2.3 * sin(12 * pi * x)
        1.3 * sin(16 * pi * x)
     ]
then the result of 
v1 + v2 = [ 1.5 * sin(2 * pi * x) +
            3.0 * sin(4 * pi * x) +
            2.0 * sin(10 * pi * x) +
            2.3 * sin(12 * pi * x) +
            1.3 * sin(16 * pi * x)
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

__all__ = ['AlgebraDict', 'Element', 'H1UIElement', 'H1UIDelta', 'H1UIAvg', 'H1UISin', 'H1UIPoly', 'Vector', 'FuncVector']

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

    def _normaliser(self, p):
        pass

    def _delta_dot(self, right, left_params, right_params):
        """ Dotting with a delta function is always the same... """
        x0 = right_params.keys_array()
        rc = right_params.values_array()
        return (right._normaliser(x0)[:,np.newaxis] * rc[:,np.newaxis] * self.evaluate(left_params, x0)).sum()

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

class H1UISin(H1UIElement):

    def evaluate(self, params, x):
        m = params.keys_array()
        c = params.values_array()
        return c * np.sin(math.pi * np.outer(x, m)) * self._normaliser(m)
    
    def dot(self, right, left_params, right_params):
        lc = left_params.values_array()
        rc = right_params.values_array()
        lp = left_params.keys_array()
        rp = left_params.keys_array()

        if isinstance(right, H1UIDelta):
            return self._delta_dot(right, left_params, right_params)
        elif isinstance(right, H1UIAvg):
            return self._avg_dot(right, left_params, right_params)
        elif isinstance(right, H1UISin):
            return (lc[:,np.newaxis] * rc * np.equal.outer(lp, rp)).sum()
        #elif isinstance(right, (H1UIAvg, H1UIPoly)):
        return 0.0 # TO BE IMPLEMENTED

    def _avg_dot(self, right, left_params, right_params):
        a = right_params.keys_array()[:, 0]
        b = right_params.keys_array()[:, 1]
        rc = right_params.values_array()

        m = left_params.keys_array()
        lc = left_params.values_array()
        
        ln = self._normaliser(left_params.keys_array())
        rn = right._normaliser(right_params.keys_array())
         
        return (ln[:, np.newaxis] * rn * lc[:,np.newaxis] * rc * (np.cos(math.pi * m[:,np.newaxis] * a) \
                - np.cos(math.pi * m[:,np.newaxis] * b)) / (math.pi * m[:,np.newaxis] * (b - a))).sum()
    
    def _normaliser(self, p):
        return math.sqrt(2.0) / (math.pi * p)

class H1UIDelta(H1UIElement):

    def evaluate(self, params, x):
        # nb we allow both x0 and x to be np arrays
        # returns an array of size len(x) * len(x0)
        x0 = params.keys_array()
        c = params.values_array()

        # This is now a matrix of size len(x) * len(x0)
        choice = np.less.outer(x, x0) #np.array([x > x0ref for x0ref in x0])

        lower = self._normaliser(x0) * np.outer(x, (1. - x0))
        upper = self._normaliser(x0) * np.outer((1. - x), x0)
        
        if np.isscalar(choice):
            return lower * choice + upper * (not choice)

        return c * (lower * choice + upper * (~choice))
    
    def dot(self, right, left_params, right_params):
        if isinstance(right, H1UIDelta):
            return self._delta_dot(right, left_params, right_params)
        elif isinstance(right, (H1UISin, H1UIAvg, H1UIPoly)):
            return right.dot(self, right_params, left_params)
        return 0.0

    def _normaliser(self, p):
        return 1. / np.sqrt((1. - p) * p)

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
        
        return c * self._normaliser(params.keys_array()) * (low * l + m * mid + h * hi)
        
    def dot(self, right, left_params, right_params):
        if isinstance(right, H1UIDelta):
            return self._delta_dot(right, left_params, right_params)
        elif isinstance(right, H1UIAvg):
            return self._avg_dot(right, left_params, right_params)
        elif isinstance(right, (H1UISin, H1UIPoly)):
            return right.dot(self, right_params, left_params)
        return 0.0
       
    
    def _avg_dot(self, right, left_params, right_params):
        a = left_params.keys_array()[:,0][:,np.newaxis]
        b = left_params.keys_array()[:,1][:,np.newaxis]
        lc = left_params.values_array()[:,np.newaxis]
        ln = self._normaliser(left_params.keys_array())[:,np.newaxis]

        c = right_params.keys_array()[:,0]
        d = right_params.keys_array()[:,1]
        rc = right_params.values_array()
        rn = right._normaliser(right_params.keys_array())

        if any(a > b) or any(c > d): 
            raise Exception('Some local-average intervals are in reverse, a > b')

        dot = (b <= c) * self._disj(a, b, c, d)
        dot += ((a < c) & (c <= b) & (b <= d)) * self._intr(a, b, c, d)
        dot += ((a <= c) & (d <= b)) * self._cont(a, b, c, d)
        dot += ((c < a) & (b < d)) * self._cont(c, d, a, b)
        dot += ((c < a) & (a <= d) & (d < b)) * self._intr(c, d, a, b)  
        dot += (d < a) * self._disj(c, d, a, b)

        return (lc * ln * rc * rn * dot).sum()
        

    def _avg_dot_slow(self, right, left_params, right_params):
        # Oooh this one's a bit tricky eh
        
        # Lets implement a loop version first. 
        ab = left_params.keys_array()
        lc = left_params.values_array()
        cd = right_params.keys_array()
        rc = right_params.values_array()
        
        if any(ab[:,0] > ab[:,1]) or any(cd[:,0] > cd[:,1]):
            raise Exception('Some local-average intervals are in reverse, a > b')

        dot = 0.0
        # Note: might have to .T The arrays... depending
        for a, b in zip(*ab.T):
            for c, d in zip(*cd.T):
                
                # There are exactly 6 possible cases
                if b <= c:
                    dot += self._disj(a, b, c, d) * (self._nt(a, b) * self._nt(c, d))
                elif a < c and c <= b and b <= d:
                    dot += self._intr(a, b, c, d) * (self._nt(a, b) * self._nt(c, d))
                elif a <= c and d <= b:
                    dot += self._cont(a, b, c, d) * (self._nt(a, b) * self._nt(c, d))
                elif c <= a and b <= d:
                    dot += self._cont(c, d, a, b) * (self._nt(a, b) * self._nt(c, d))
                elif c < a and a <= d and d <= b:
                    dot += self._intr(c, d, a, b) * (self._nt(a, b) * self._nt(c, d))
                elif c <= d and d < a:
                    dot += self._disj(c, d, a, b) * (self._nt(a, b) * self._nt(c, d))
                else:
                    raise Exception('Uh oh, something went wrong')
        return dot
   
    # These internal functions represent the different cases for when the local avg is dotted against itself
    def _disj(self, a, b, c, d):
        return (1.0 - 0.5 * (c + d)) * 0.5 * (a + b)
    def _intr(self, a, b, c, d):
        return (1.0 - 0.5 * (c + d)) * 0.5 * (a + b) - (b - c)**3 / (6.0 * (b - a) * (d - c))
    def _cont(self, a, b, c, d):
        return (1.0/(b-a)) * ((1 - 0.5 * (c + d)) * 0.5 * (d*d - a*a) - (d - c)*(d - c) / 6.0 \
                - 0.25 * (c + d) * ((1-b)*(1-b) - (1-d)*(1-d)))

    def _normaliser(self, p):
        # p is assumed to be an array of size n*2
        return 1.0 / np.sqrt(p[:,0] + (p[:,1] - p[:,0])/3.0 - 0.25 * (p[:,0] + p[:,1]) * (p[:,0] + p[:,1]))

    def _nt(self, a, b):
        return 1.0 / np.sqrt(a + (b - a)/3.0 - 0.25 * (a + b) * (a + b))

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

    __rsub__ = __sub__

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

    def __mul__(self, other):
        result = copy.deepcopy(self)
        result.elements *= other
        return result

    __rmul__ = __mul__

    def __truediv__(self, other):
        result = copy.deepcopy(self)
        result.elements /= other
        return result


