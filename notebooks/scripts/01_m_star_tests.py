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

N = 1e4
dictionary = pat.make_unif_dictionary(N)

np.random.seed(3)

ns = range(40,201,2)

ms_comp = np.zeros(len(ns))
ms_wcomp = np.zeros(len(ns))

Vn = pat.make_sin_basis(ns[-1])

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

np.savetxt('comp_sin_m_star', BP_comp)
np.savetxt('wcomp_sin_m_star', BP_wcomp)

