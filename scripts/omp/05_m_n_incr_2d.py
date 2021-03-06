import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys, os
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

try:
    n_min = int(sys.argv[1])
    n_max = int(sys.argv[2])
    n_step = int(sys.argv[3])
except IndexError:
    print("Usage: " + sys.argv[0] + " n_min n_max n_step")
    sys.exit(1)

ns = range(n_min,n_max,n_step)
beta_star = 0.5

N = 1e3
dictionary = pat.make_unif_dictionary(N)

np.random.seed(3)

ms_comp = np.zeros((len(ns),2), dtype=np.int16)
ms_wcomp = np.zeros((len(ns),2), dtype=np.int16)

Vn = pat.make_sin_basis(ns[-1])

cgbc = pat.CollectiveOMP(dictionary, Vn.subspace(slice(0,ns[0])), verbose=True)
Wm_comp = cgbc.construct_to_beta(beta_star)
ms_comp[0,0] = ns[0]
ms_comp[0,1] = cgbc.m

wcgbc = pat.WorstCaseOMP(dictionary, Vn.subspace(slice(0,ns[0])), verbose=True)
Wm_wcomp = wcgbc.construct_to_beta(beta_star)
ms_wcomp[0,0] = ns[0]
ms_wcomp[0,1] = wcgbc.m

for j, n in enumerate(ns[1:]):
    
    cgbc.Vn = Vn.subspace(slice(0,n))
    Wm_comp = cgbc.construct_to_beta(beta_star)
    ms_comp[j,0] = n
    ms_comp[j,1] = cgbc.m
    
    wcgbc.Vn = Vn.subspace(slice(0,n))
    Wm_wcomp = wcgbc.construct_to_beta(beta_star)
    ms_wcomp[j,0] = n 
    ms_wcomp[j,1] = wcgbc.m

comp_file = './comp_sin_n_incr_m_star.csv'
if os.path.isfile(comp_file):
    ms_comp_prev = np.loadtxt(comp_file)
    ms_comp = np.append(ms_comp_prev, ms_comp, axis=0)
np.savetxt(comp_file, ms_comp, fmt='%i')

wcomp_file = './wcomp_sin_n_incr_m_star.csv'
if os.path.isfile(wcomp_file):
    ms_wcomp_prev = np.loadtxt(wcomp_file)
    ms_wcomp = np.append(ms_wcomp_prev, ms_wcomp, axis=0)
np.savetxt(wcomp_file, ms_wcomp, fmt='%i')

