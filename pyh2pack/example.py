import pyh2pack
import numpy as np

'''
   NOTE: 
   In Jupyter notebook, the outputs of `print_statistics/print_setting' might be redirected to terminals and will not be properly shown. 
   Solution to this problem is to use package 'wurlitzer'   
   Run `%load_ext wurlitzer` in Jupyeter. 
'''

N = 80000
krnl_dim = 1
pt_dim = 2
coord = np.random.uniform(0, 1, size=(pt_dim, N))
x = np.random.normal(size=(krnl_dim*N))


'''
   Test without precomputed proxy points
'''
#   build
krnl_param = np.array([1, -0.5])
A = pyh2pack.H2Mat(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=pt_dim, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param)
#   matvec
y = A.matvec(x)
#   partial direct matvec
start_pt = 8000
end_pt = 9999
z = A.direct_matvec(x, start_pt, end_pt)
#   print the matvec error in the partial results
print(np.linalg.norm(y[start_pt*krnl_dim:(end_pt+1)*krnl_dim] - z) / np.linalg.norm(z))
#   statistic info of pyh2pack performance
A.print_statistic()
A.print_setting()
A.clean()



'''
   Test with precomputed proxy points
'''
#   path to the file of storing proxy points
pp_fname = "./pp_tmp.dat"
#   build
krnl_param = np.array([1,-0.5])
A = pyh2pack.H2Mat(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=pt_dim, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param, pp_filename=pp_fname)
#   matvec
y = A.matvec(x)
#   partial direct matvec
start_pt = 8000
end_pt = 9999
z = A.direct_matvec(x, start_pt, end_pt)
#   print the matvec error in the partial results
print(np.linalg.norm(y[start_pt*krnl_dim:(end_pt+1)*krnl_dim] - z) / np.linalg.norm(z))
#   statistic info of pyh2pack performance
A.print_statistic()
A.clean()


'''
   Test with matmul
'''
#   build
krnl_param = np.array([1,-0.5])
A = pyh2pack.H2Mat(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=pt_dim, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param)
#   matmul 
nvec = 10
xs = np.random.normal(size=(krnl_dim*N, nvec))
ys = A.matmul(xs)
#  partial direct sum
zs = []
start_pt = 0
end_pt = 999
for i in range(nvec):
   zs.append(A.direct_matvec(xs[:,i], start_pt, end_pt))
zs = np.hstack([z[:,np.newaxis] for z in zs])
print(np.linalg.norm(ys[start_pt*krnl_dim:(end_pt+1)*krnl_dim, :] - zs, ord='fro') / np.linalg.norm(zs,  ord='fro'))
A.print_statistic()
A.clean()



'''
   Test with direct matrix vector multiplication in pyh2pack
'''
A = pyh2pack.H2Mat(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=pt_dim, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param)
y = A.matvec(x)

#   partial direct matvec by class h2 variable.
start_pt = 0
end_pt = 999
z = A.direct_matvec(x, start_pt, end_pt)

#   direct matvec via package method: kernel_matvec
target_coord = coord[:, start_pt:(end_pt+1)]
z0 = pyh2pack.kernel_matvec(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_dim=pt_dim, krnl_param=krnl_param, source=coord, target=target_coord, x_in=x)

A_blk = pyh2pack.kernel_block(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_dim=pt_dim, krnl_param=krnl_param, source=coord, target=target_coord)
z1 = np.matmul(A_blk, x)

#  check error
print(np.linalg.norm(z - z0))
print(np.linalg.norm(z - z1))
A.print_statistic()
A.clean()