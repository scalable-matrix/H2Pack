import pyh2pack
import numpy as np

N = 80000
krnl_dim = 1
pt_dim = 3
coord = np.random.uniform(0, 1, size=(pt_dim, N))
x = np.random.normal(size=(krnl_dim*N))

# build
krnl_param = np.array([0.5])
A = pyh2pack.H2Mat(kernel="Gaussian_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=pt_dim, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param, sample_pt=1)
# Coulomb kernel does not have krnl_param
#A = pyh2pack.H2Mat(kernel="Coulomb_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=pt_dim, JIT_mode=1, rel_tol=1e-6, sample_pt=1)

# show build settings
A.print_setting()

# matvec
y = A.matvec(x)

# partial direct matvec
start_pt = 8000
end_pt = 9999
z = A.direct_matvec(x, start_pt, end_pt)

# print the matvec relative error in the partial results
relerr = np.linalg.norm(y[start_pt*krnl_dim:(end_pt+1)*krnl_dim] - z) / np.linalg.norm(z)
print("H2 matvec relative error = %e\n" % relerr)

# statistic info of pyh2pack performance
A.print_statistic()

# clean out
A.clean()
