"""
utils.py

Author: James Ashton Nichols
Start date: June 2017

This contains a few utilities that are useful in constructing various bases,
from random point evals to complete sinusoidal bases, and various other situations. 
"""
import numpy as np

from pyApproxTools.vector import *
from pyApproxTools.basis import *

__all__ = ['make_sin_basis', 'make_random_delta_basis', 'make_random_avg_basis', 'make_unif_avg_basis', 'make_unif_dictionary', 'make_rand_dictionary', 'make_unif_avg_dictionary']

def make_sin_basis(n):
    V_n = []

    # We want an ordering such that we get (1,1), (1,2), (2,1), (2,2), (2,3), (3,2), (3,1), (1,3), ...
    for i in range(1,n+1):
        v_i = FuncVector(params=[[i]], coeffs=[[1.0]], funcs=['H1UISin'])
        V_n.append(v_i)
            
    return Basis(V_n, is_orthonormal=True)

def make_random_delta_basis(n, bounds=None, bound_prop=1.0):

    vecs = []
    
    if bounds is not None:
        bound_points = (bounds[1] - bounds[0]) *  np.random.random(round(n * bound_prop)) + bounds[0]
        
        remain_points = (1.0 - (bounds[1] - bounds[0])) * np.random.random(round(n * (1.0 - bound_prop)))
        # Ooof remain points problem - first left
        if bounds[0] > 0.0:
            remain_l = remain_points[remain_points < bounds[0]]
            remain_r = remain_points[remain_points >= bounds[0]] + bounds[1]
            remain_points = np.append(remain_l, remain_r)
        else:
            remain_points += bounds[1]

        points = np.append(bound_points, remain_points)
    else:
        points = np.random.random(n)
        
    for i in range(n):
        v_i = FuncVector(params=[[points[i]]], coeffs=[[1.0]], funcs=['H1UIDelta']) 
        vecs.append(v_i)
    
    return Basis(vecs)

def make_random_avg_basis(n, epsilon=1.0e-2, bounds=None, bound_prop=1.0):

    vecs = []
    
    if bounds is not None:
        bound_points = (bounds[1] - bounds[0]) *  np.random.random(round(n * bound_prop)) + bounds[0]
        
        remain_points = (1.0 - (bounds[1] - bounds[0])) * np.random.random(round(n * (1.0 - bound_prop)))
        # Ooof remain points problem - first left
        if bounds[0] > 0.0:
            remain_l = remain_points[remain_points < bounds[0]]
            remain_r = remain_points[remain_points >= bounds[0]] + bounds[1]
            remain_points = np.append(remain_l, remain_r)
        else:
            remain_points += bounds[1]

        points = np.append(bound_points, remain_points)
    else:
        points = np.random.random(n)
       
    # We need to contract the points by epsilon on both sides...
    points = (1.0 - epsilon) * points + epsilon

    for i in range(n):
        v_i = FuncVector(params=[[(points[i]-0.5*epsilon, points[i]+0.5*epsilon)]], coeffs=[[1.0]], funcs=['H1UIAvg']) 
        vecs.append(v_i)
    
    return Basis(vecs)

def make_unif_avg_basis(m, epsilon):

    return Basis(make_unif_avg_dictionary(m, epsilon))

def make_unif_dictionary(N):

    points, step = np.linspace(0.0, 1.0, N+1, endpoint=False, retstep=True)
    #points = points + 0.5 * step # Make midpoints... don't want 0.0 or 1.0
    points = points[1:] # Get rid of that first one!

    dic = [FuncVector(params=[[p]],coeffs=[[1.0]],funcs=['H1UIDelta']) for p in points]

    return dic

def make_unif_avg_dictionary(N, epsilon):

    points = np.linspace(0.5 * epsilon, 1.0 - 0.5 * epsilon, N, endpoint=True)
    #points = points + 0.5 * step # Make midpoints... don't want 0.0 or 1.0

    dic = [FuncVector(params=[[(p-0.5*epsilon, p+0.5*epsilon)]],coeffs=[[1.0]],funcs=['H1UIAvg']) for p in points]

    return dic

def make_rand_dictionary(N):

    points = np.random.random(N)

    dic = [FuncVector(params=[[p]],coeffs=[[1.0]],funcs=['H1UIDelta']) for p in points]

    return dic

