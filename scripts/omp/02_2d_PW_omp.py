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

np.random.seed(3)

fem_div = 7
field_div = 2

n = 20
m = 200

try:
    width = int(sys.argv[1])
except IndexError:
    print("Usage: " + sys.argv[0] + " width")
    sys.exit(1)

# Create Vn - an orthonormalised reduced basis
Vn, fields = pat.make_pw_reduced_basis(n, field_div=field_div, fem_div=fem_div)
Vn = Vn.orthonormalise()

Wms_c = []
Wms_wc = []

bs_c = np.zeros(m) 
bs_wc = np.zeros(m) 

print('Construct dictionary of local averages...')
D = pat.make_pw_hat_rep_dict(fem_div, width=width)

print('Greedy basis construction...')
cbc = pat.CollectiveOMP(D, Vn, Wm=pat.PWBasis(), verbose=True)
Wm_c = cbc.construct_to_m(m)
Wm_c_o = Wm_c.orthonormalise()
Wms_c.append(Wm_c)
Wm_c_o.save('Wm_c_{0}'.format(width))

wcbc = pat.WorstCaseOMP(D, Vn, Wm=pat.PWBasis(), verbose=True)
Wm_wc = wcbc.construct_to_m(m)
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

    bs_c[i] = BP_c.beta()
    bs_wc[i] = BP_wc.beta()

np.save('bs_c_{0}'.format(width), bs_c)
np.save('bs_wc_{0}'.format(width), bs_wc)
