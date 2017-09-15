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

try:
    n_min = int(sys.argv[1])
    n_max = int(sys.argv[2])
    n_step = int(sys.argv[3])
except IndexError:
    print("Usage: " + sys.argv[0] + " n_min n_max n_step")
    sys.exit(1)

ns = range(n_min,n_max+1,n_step)
print(ns)
beta_star = 0.5

N = 1e3
dictionary = pat.make_unif_dictionary(N)

np.random.seed(3)

ms_comp = np.zeros((2,len(ns)), dtype=np.int16)
ms_wcomp = np.zeros((2,len(ns)), dtype=np.int16)

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
            ms_comp[j,0] = n #BP_comp.beta()
            ms_comp[j,1] = i #BP_comp.beta()
            break

    for i in range(n, m):
        BP_wcomp =  BP_wcomp_l.subspace(Wm_indices=slice(0,i))
        if BP_wcomp.beta() > beta_star:
            ms_wcomp[j,0] = n #BP_wcomp.beta()
            ms_wcomp[j,1] = i #BP_wcomp.beta()
            break

np.savetxt('comp_sin_m_star.csv', ms_comp, fmt='%i')
np.savetxt('wcomp_sin_m_star.csv', ms_wcomp, fmt='%i')

