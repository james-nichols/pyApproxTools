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

fem_div = 7
field_div = 2

n = 20
m = 100

widths = [2**i for i in range(fem_div-3)]
print(widths)

# Create Vn - an orthonormalised reduced basis
Vn, fields = pat.make_pw_reduced_basis(n, field_div=2, fem_div=fem_div)
Vn = Vn.orthonormalise()

Wms_c = []
Wms_wc = []

bs_c = np.zeros((len(widths), m))
bs_wc = np.zeros((len(widths), m))

for j, width in enumerate(widths):

    print('Construct dictionary of local averages...')
    D = pat.make_pw_hat_rep_dict(fem_div, width=width)

    print('Greedy basis construction...')
    cbc = pat.CollectiveOMP(m, D, Vn, Wm=pat.PWBasis(), verbose=True)
    Wm_c = cbc.construct_basis()
    Wm_c_o = Wm_c.orthonormalise()
    Wms_c.append(Wm_c)
    Wm_c_o.save('Wm_c_{0}'.format(width))

    wcbc = pat.WorstCaseOMP(m, D, Vn, Wm=pat.PWBasis(), verbose=True)
    Wm_wc = wcbc.construct_basis()
    Wm_wc_o = Wm_wc.orthonormalise()
    Wms_wc.append(Wm_wc)
    Wm_wc_o.save('Wm_wc_{0}'.format(width))

    # For efficiency it makes sense to compute the basis pair and the associated
    # cross-gramian only once, then sub sample it as we grow m...
    BP_c_l = pat.BasisPair(Wm_c_o, Vn)
    BP_wc_l = pat.BasisPair(Wm_wc_o, Vn)

    for i in range(n, m):
        BP_c =  BP_c_l.subspace(Wm_indices=slice(0,i))
        BP_wc =  BP_wc_l.subspace(Wm_indices=slice(0,i))

        bs_c[j, i] = BP_c.beta()
        bs_wc[j, i] = BP_wc.beta()

np.save('bs_c', bs_c)
np.save('bs_wc', bs_wc)
