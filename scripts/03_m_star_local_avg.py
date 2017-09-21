import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

beta_star = 0.5

N = 1e3

eps_1 = 0.025
eps_2 = 0.01
eps_3 = 0.0025

dictionaries = [pat.make_unif_dictionary(N), pat.make_unif_avg_dictionary(N, eps_1),\
                pat.make_unif_avg_dictionary(N, eps_2), pat.make_unif_avg_dictionary(N, eps_3)]

np.random.seed(3)

ns = range(40,51,2)

ms_comp = np.zeros(len(dictionaries), len(ns))
ms_wcomp = np.zeros(len(dictionaries), len(ns))

Vn = pat.make_sin_basis(ns[-1])
for k, dictionary in enumerate(dictionaries):
    
    for j, n in enumerate(ns):
        
        m = 2 * n

        gbc = pat.CollectiveOMP(m, dictionary, Vn.subspace(slice(0,n)), verbose=True)
        Wm_comp = gbc.construct_basis()
        Wm_comp_o = Wm_comp.orthonormalise()

        wcgbc = pat.WorstCaseOMP(m, dictionary, Vn.subspace(slice(0,n)), verbose=True)
        Wm_wcomp = wcgbc.construct_basis()
        Wm_wcomp_o = Wm_wcomp.orthonormalise()

        BP_comp_l = pat.BasisPair(Wm_comp_o, Vn.subspace(slice(0,n)))
        BP_wcomp_l = pat.BasisPair(Wm_wcomp_o, Vn.subspace(slice(0,n)))    
        
        for i in range(n, m):
            BP_comp = BP_comp_l.subspace(Wm_indices=slice(0,i))
            if BP_comp.beta() > beta_star:
                ms_comp[j] = i #BP_comp.beta()
                break

        for i in range(n, m):
            BP_wcomp =  BP_wcomp_l.subspace(Wm_indices=slice(0,i))
            if BP_wcomp.beta() > beta_star:
                ms_wcomp[j] = i #BP_wcomp.beta()
                break

np.savetxt('comp_sin_avg_m_star', BP_comp)
np.savetxt('wcomp_sin_avg_m_star', BP_wcomp)

