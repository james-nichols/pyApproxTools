import numpy as np
import scipy as sp

import sys
sys.path.append("../../")
import pyApproxTools as pat

def make_soln(points, fem_div, field_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        field = pat.PWConstantSqDyadicL2(a_bar + c * p.reshape((2**field_div,2**field_div)))
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields

fem_div = 7

a_bar = 1.0
c = 0.9
field_div = 2
side_n = 2**field_div

n_us = 40

np.random.seed(3)
points = 2*np.random.random((n_us, side_n*side_n)) - 1

us, fields = make_soln(points, fem_div, field_div, a_bar=a_bar, c=c, f=1.0)


# local_width is the width of the measurement squares in terms of FEM mesh squares
width_div = 1
local_width = 2**width_div
spacing_div = 4

Wm_reg, Wloc_reg = pat.make_local_avg_grid_basis(width_div, spacing_div, fem_div, return_map=True)
Wm_reg = Wm_reg.orthonormalise()

m = Wm_reg.n

Wm_rand, Wloc = pat.make_pw_local_avg_random_basis(m=m, div=fem_div, width=local_width, return_map=True)
Wm_rand = Wm_rand.orthonormalise()

N = int(1e4)
dict_basis, dict_fields = pat.make_pw_reduced_basis(N, field_div, fem_div, a_bar=a_bar, c=c, f=1.0, verbose=False)
dictionary = dict_basis.vecs

Vn_sin = pat.make_pw_sin_basis(div=fem_div, N=8)
Vn_red, Vn_red_fields = pat.make_pw_reduced_basis(m, field_div, fem_div, a_bar=a_bar, c=c, f=1.0, verbose=False)

g = pat.GreedyApprox(dictionary, Vn=pat.PWBasis(), verbose=False)
g.construct_to_n(m)

for Vn, label in zip(generic_Vns, generic_Vns_labels):
    Vn.save('results/' + label + '_Basis')

import copy

adapted_Vns = []
adapted_Vns_labels = ['MBOMP', 'MBPP']

for Wm, Wm_label in zip([Wm_reg, Wm_rand], ['Reg', 'Rand']):

    algs = [pat.MeasBasedOMP(dictionary, us[0], Wm, Vn=pat.PWBasis(), verbose=True), 
            pat.MeasBasedPP(dictionary, us[0], Wm, Vn=pat.PWBasis(), verbose=True)]
    
    adapted_Vns.append([])
    adapted_Vns_labels.append([])
    
    for alg, Vn_label in zip(algs, adapted_Vns_labels):
        adapted_Vns[-1].append([])
        for i, u in enumerate(us):
            if i > 0:
                alg.reset_u(u)

            alg.construct_to_n(m)

            alg.Vn.save('results/' + Vn_label + '_' + Wm_label + '_{0}_Basis'.format(i))
            adapted_Vns[-1][-1].append(alg.Vn) 


N = int(1e3)
np.random.seed(1)
dict_basis_small, dict_fields = pat.make_pw_reduced_basis(N, field_div, fem_div, a_bar=a_bar, c=c, f=1.0, verbose=False)
dict_basis_small.make_grammian()

cent = dict_basis_small.reconstruct(np.ones(N) / N)

import copy
cent_vecs = copy.deepcopy(dict_basis_small.vecs)
for i in range(len(cent_vecs)):
    cent_vecs[i] = cent_vecs[i] - cent

dict_basis_small_cent = pat.PWBasis(cent_vecs)
dict_basis_small_cent.make_grammian()

lam, V = np.linalg.eigh(dict_basis_small_cent.G)
PCA_vecs = [cent]
for i, v in enumerate(np.flip(V.T, axis=0)[:m]):
    vec = dict_basis_small_cent.reconstruct(v)
    PCA_vecs.append(vec / vec.norm())

Vn_PCA = pat.PWBasis(PCA_vecs)


generic_Vns = [Vn_sin, Vn_red, g.Vn, Vn_PCA]
generic_Vns_labels = ['Sinusoid', 'Reduced', 'PlainGreedy', 'PCA']

stats = np.zeros([6, 2, 6, n_us, m]) # 6 stats, 2 Wms, 5 Vns (sin, red, greedy, omp, pp)

for i, (Wm, Wm_label) in enumerate(zip([Wm_reg, Wm_rand], ['Reg grid', 'Random'])):
    
    for j, g in enumerate(generic_Vns):
        
        Vn_big = g.orthonormalise()
        
        for l, n in enumerate(range(2,min(Vn_big.n, m))):
            
            Vn = Vn_big.subspace(slice(0,n))
            BP = pat.BasisPair(Wm, Vn)

            stats[2, i, j, :, n] = BP.beta()

            for k, u in enumerate(us):
                u_p_v = Vn.project(u)
                u_star, v_star, w_p, v_w_p, cond = BP.measure_and_reconstruct(u)
    
                stats[0, i, j, k, n] = (u - u_star).norm()
                stats[1, i, j, k, n] = (u - u_p_v).norm()
                stats[3, i, j, :, n] = cond
                stats[4, i, j, :, n] = (u_star - v_star).norm()
    
    for j_i, a in enumerate(adapted_Vns[i]):
        j = j_i + len(generic_Vns)
        for k, u in enumerate(us):
            Vn_big = a[k].orthonormalise()
            
            for l, n in enumerate(range(2,min(Vn_big.n, m))):
            
                Vn = Vn_big.subspace(slice(0,n))

                u_p_v = Vn.project(u)
                BP = pat.BasisPair(Wm, Vn)
                u_star, v_star, w_p, v_w_p, cond = BP.measure_and_reconstruct(u)

                stats[0, i, j, k, n] = (u - u_star).norm()
                stats[1, i, j, k, n] = (u - u_p_v).norm()
                stats[2, i, j, k, n] = BP.beta()
                stats[3, i, j, k, n] = cond
                stats[4, i, j, k, n] = (u_star - v_star).norm()

np.save('results/07_greedy_Vn_stats', stats)
