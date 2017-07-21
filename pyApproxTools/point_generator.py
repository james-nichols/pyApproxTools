"""
point_generator.py

Author: James Ashton Nichols
Start date: Some time in 2014?

A Python 3 adaptation of my old QMC code from my PhD thesis. Does point set generation,
Monte Carlo and quasi-Monte Carlo
"""

import numpy as np
import math
from scipy.stats import norm

def inversenormal(z):
    #znor = np.zeros(len(z))
    #for i in range(len(z)):
        # norm.ppf is the python implementation of 
        # the cumulative-normal inverse
    #    znor[i] = norm.ppf(z[i])
    znor = norm.ppf(z)
    return znor

def inversenormal_fast(p):
    """
    Modified from the author's original perl code (original comments follow below)
    by dfield@yahoo-inc.com.  May 3, 2004.

    Lower tail quantile for standard normal distribution function.

    This function returns an approximation of the inverse cumulative
    standard normal distribution function.  I.e., given P, it returns
    an approximation to the X satisfying P = Pr{Z <= X} where Z is a
    random variable from the standard normal distribution.

    The algorithm uses a minimax approximation by rational functions
    and the result has a relative error whose absolute value is less
    than 1.15e-9.

    Author:      Peter John Acklam
    Time-stamp:  2000-07-19 18:26:14
    E-mail:      pjacklam@online.no
    WWW URL:     http://home.online.no/~pjacklam

    MODIFIED BY James Ashton Nichols to use NumPy libraries for fast vector operations, 2012-10-14
    """

    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01,  2.209460984245205e+02, \
         -2.759285104469687e+02,  1.383577518672690e+02, \
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, \
         -1.556989798598866e+02,  6.680131188771972e+01, \
         -1.328068155288572e+01 )
    c = (-7.784894002430293e-03, -3.223964580411365e-01, \
         -2.400758277161838e+00, -2.549732539343734e+00, \
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01, \
          2.445134137142996e+00,  3.754408661907416e+00)

    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow

    # Rational approximation for central region:
    q = p - 0.5
    r = q*q
    result = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    
    try:
        for i in range(len(p)):
            if p[i] < plow:
                # Rational approximation for lower region:
                q  = math.sqrt(-2*math.log(p[i]))
                result[i] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
            elif phigh < p[i]:
                # Rational approximation for upper region:
                q  = math.sqrt(-2*math.log(1-p[i]))
                result[i] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    except TypeError:
        if p < plow:
            # Rational approximation for lower region:
            q  = math.sqrt(-2*math.log(p))
            result = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        elif phigh < p:
            # Rational approximation for upper region:
            q  = math.sqrt(-2*math.log(1-p))
            result = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        
    return result

def shift_point(self, s, x):
    
    xdelta = np.zeros(length(x))
    
    for i in range(length(x)):
        tmp = x[i] + s[i]
        xdelta[i] = tmp - np.floor(tmp)
        if xdelta[i] == 0.0:
            xdelta[i] = 1e-32

    return xdelta

class PointGenerator(object):

    def __init__(self, d, n, lims=None):

        self.d = d
        self.n = n

        self.points = None
        self.count = 0

        if lims is not None:
            self.lims = lims

    def next_point(self):
        count += 1
        return self.points[count-1, :] # Because we start the count at zeros

    def shift_point(self, shift):
        point_shifted = self.points[:, self.count] + shift
        return point_shifted - np.floor(point_shifted)

    def shift_all_points(self, shift):
        point_shifted = self.points + shift
        return point_shifted - np.floor(point_shifted)

    def apply_random_shift(self):
        shift = np.random.random(d)
        self.points = self.points + shift

class MonteCarlo(PointGenerator):

    def __init__(self, d, n, lims=None, seed=None):
        
        super().__init__(d, n, lims)

        if seed is not None:
            np.random.seed(seed)

        if lims is not None and len(lims) == 2:
            self.points = (lims[1] - lims[0]) * np.random.random((n, d)) + lims[0]
        else: 
            self.points = np.random.random(n, d)

class QMCLatticeRule(PointGenerator):

    def __init__(self, d, n, lims=None, filename=None):

        super().__init__(d, n, lims)

        self.z = np.zeros(self.d)
        if filename is not None:
            self.get_generator(filename)
        else:
            self.get_generator('lattice_rules/lattice-32001-1024-1048576.3600')

        self.points = np.modf(np.outer(self.z, np.arange(n)/n))[0].T

        if lims is not None and len(lims) == 2:
            self.points = (lims[1] - lims[0]) * self.points + lims[0]

    def get_generator(self, filename):
        gen_file = open(filename, 'r')
        
        count = 0
        for line in gen_file:
            l = line.split()
            self.z[count] = l[1]
            count += 1
            if count >= self.d:
                break

        gen_file.close()

