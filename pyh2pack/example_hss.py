import pyh2pack
import numpy as np

'''
   NOTE: 
   In Jupyter notebook, the outputs of `print_statistics/print_setting' might be redirected to terminals and will not be properly shown. 
   Solution to this problem is to use package 'wurlitzer'   
   Run `%load_ext wurlitzer` in Jupyeter. 
'''

N = 40000
krnl_dim = 1
pt_dim = 2
coord = np.random.uniform(0, N**(1.0/pt_dim), size=(pt_dim, N))
x = np.random.normal(size=(krnl_dim*N))

'''
Standard HSS 
'''

##   build
krnl_param = np.array([1, -0.5])
A = pyh2pack.HSSMat(kernel="Quadratic_2D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=pt_dim, rel_tol=1e-3, krnl_param=krnl_param)
# A = pyh2pack.HSSMat(kernel="Quadratic_2D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=pt_dim, rank=100, krnl_param=krnl_param)


##   matvec  
y = A.matvec(x)
#   partial direct matvec
start_pt = 6000
end_pt = 9999
z = A.direct_matvec(x, start_pt, end_pt)
#   print the matvec error in the partial results
print(np.linalg.norm(y[start_pt*krnl_dim:(end_pt+1)*krnl_dim] - z) / np.linalg.norm(z))

##  ULV factorization
diag_shift = 0.1
A.factorize(is_cholesky=1, shift=diag_shift)

##   solve based on ULV decomposition
b = y + diag_shift * x
x0 = A.solve(b)
print("HSS solve error (compared to HSS matvec) %.3e" % (np.linalg.norm(x - x0) / np.linalg.norm(x)))

##   partial solve, A = LU, apply inv(L) first and then apply inv(U)
z = A.solve(b, op="L")
x1 = A.solve(z, op="U")
print("HSS solve error (compared to HSS matvec) %.3e" % (np.linalg.norm(x - x1) / np.linalg.norm(x)))

##   statistic info of pyh2pack performance
A.print_statistic()
A.print_setting()
A.clean()







'''
SPD HSS 
'''

##   build
krnl_param = np.array([1, -0.5])
A = pyh2pack.HSSMat("Quadratic_2D", krnl_dim, coord, pt_dim, rel_tol=1e-6, krnl_param=krnl_param, spdhss=1, spdhss_shift=0.0, rank=100)
# A = pyh2pack.HSSMat(kernel="Quadratic_2D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=pt_dim, rank=100, krnl_param=krnl_param)


##   matvec  
y = A.matvec(x)
#   partial direct matvec
start_pt = 6000
end_pt = 9999
z = A.direct_matvec(x, start_pt, end_pt)
#   print the matvec error in the partial results
print(np.linalg.norm(y[start_pt*krnl_dim:(end_pt+1)*krnl_dim] - z) / np.linalg.norm(z))

##  ULV factorization
diag_shift = 0.0
A.factorize(is_cholesky=1, shift=diag_shift)

##   solve based on ULV decomposition
b = y + diag_shift * x
x0 = A.solve(b)
print("HSS solve error (compared to HSS matvec) %.3e" % (np.linalg.norm(x - x0) / np.linalg.norm(x)))

##   partial solve, A = LU, apply inv(L) first and then apply inv(U)
z = A.solve(b, op="L")
x1 = A.solve(z, op="U")
print("HSS solve error (compared to HSS matvec) %.3e" % (np.linalg.norm(x - x1) / np.linalg.norm(x)))

##   statistic info of pyh2pack performance
A.print_statistic()
A.print_setting()
A.clean()

