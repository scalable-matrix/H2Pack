import pyh2pack
import numpy as np

N = 80000
krnl_dim = 1
coord = np.random.uniform(0, 5, size=(3, N))
krnl_param = np.array([1., -2.])
x = np.random.normal(size=(krnl_dim*N))

'''
   Test without precomputed proxy points
'''
#   build
A = pyh2pack.H2Mat(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=3, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param)
#   matvec
y = A.h2matvec(x)
#   partial direct matvec
start_pt = 8000
end_pt = 10000
z = A.direct_matvec(x, start_pt, end_pt)
#   print the matvec error in the partial results
print(np.linalg.norm(y[(start_pt-1)*krnl_dim:end_pt*krnl_dim] - z) / np.linalg.norm(z))
#   statistic info of pyh2pack performance
A.print_statistic()

A.clean()


'''
   Test with precomputed proxy points
'''
#   path to the file of storing proxy points
pp_fname = "./pp_tmp.dat"
#   build
A = pyh2pack.H2Mat(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=3, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param, pp_filename=pp_fname)
#   matvec
y = A.h2matvec(x)
#   partial direct matvec
start_pt = 8000
end_pt = 10000
z = A.direct_matvec(x, start_pt, end_pt)
#   print the matvec error in the partial results
print(np.linalg.norm(y[(start_pt-1)*krnl_dim:end_pt*krnl_dim] - z) / np.linalg.norm(z))
#   statistic info of pyh2pack performance
A.print_statistic()
A.clean()