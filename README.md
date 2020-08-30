H2Pack is a high-performance, shared-memory library for **linear-scaling matrix-vector multiplication of kernel matrices** based on their ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2) matrix representations. 
The key feature of H2Pack is the efficient ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2) matrix construction for kernel matrices using a hybrid compression method called the proxy point method. 

In the current release, H2Pack supports kernel matrices of form ![](https://latex.codecogs.com/svg.latex?K(X,%20X)) defined by a translationally-invariant, symmetric kernel function ![](https://latex.codecogs.com/svg.latex?K(x,%20y)) (e.g., Gaussian, Matern, Laplace, and Stokes kernels) and a point set ![](https://latex.codecogs.com/svg.latex?X) in low-dimensional space (i.e., 1D, 2D, and 3D).
H2Pack, written in C99, provides a C/C++ interface and an experimental Python interface. The Matlab prototype of H2Pack can be found in the [repo]().

**Main Functions:**

* ![](https://latex.codecogs.com/svg.latex?O(N)) complexity ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)  matrix representation construction for a kernel matrix (![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)-construction)
* ![](https://latex.codecogs.com/svg.latex?O(N)) complexity ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)  matrix-vector multiplication (![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)-matvec)
* ![](https://latex.codecogs.com/svg.latex?O(N)) complexity ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)  matrix-matrix  multiplication (![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)-matmul)


**References**:

* _X. Xing and E. Chow, Interpolative decomposition via proxy points for kernel matrices, SIAM Journal on Matrix Analysis and Applications, 41(1), 221–243 (2020)_
* _H. Huang, X. Xing, and E. Chow, H2Pack: High-performance H2 matrix package for kernel matrices using the proxy point method, ACM Transactions on Mathematical Software, accepted (2020)_

**Additional Features:** 

H2Pack also provides functions for constructing and applying HSS matrix representations: 

* HSS matrix representation construction for a kernel matrix using the proxy point method 
* HSS matrix-vector multiplication
* HSS matrix-matrix multiplication
* ULV decomposition of HSS matrix representation
* Direct solve of HSS matrix representation based on ULV decomposition


## Getting Started

* [Installing H2Pack](https://github.com/huanghua1994/H2Pack/wiki/Installing-H2Pack)
* [Basic Usage](https://github.com/huanghua1994/H2Pack/wiki/Basic-Usage)
* [Using and Writing Kernel Functions](https://github.com/huanghua1994/H2Pack/wiki/Using-and-Writing-Kernel-Functions) 
* [Two Running Modes for H2Pack](https://github.com/huanghua1994/H2Pack/wiki/Two-Running-Modes-for-H2Pack)
* [HSS-Related Computations](https://github.com/huanghua1994/H2Pack/wiki/HSS-Related-Computations)

## Advanced Configurations and Tools

* [Bi-Kernel Matvec (BKM) Functions](https://github.com/huanghua1994/H2Pack/wiki/Bi-Kernel-Matvec-Functions)
* [Vector Wrapper Functions for Kernel Evaluations](https://github.com/huanghua1994/H2Pack/wiki/Vector-Wrapper-Functions-For-Kernel-Evaluations)
* [Python Interface](https://github.com/huanghua1994/H2Pack/wiki/Using-H2Pack-in-Python)


## Numerical Tests

* [Accuracy Tests on Various Kernels](https://github.com/huanghua1994/H2Pack/wiki/Accuracy-Tests-on-Various-Kernels)
* [Linear Scaling Tests](https://github.com/huanghua1994/H2Pack/wiki/Linear-Scaling-Tests)
* [Parallel Efficiency Tests](https://github.com/huanghua1994/H2Pack/wiki/Parallel-Efficiency-Tests)
* [Comparative Tests on H2-matvec and H2-matmul](https://github.com/huanghua1994/H2Pack/wiki/Comparative-Tests-on-H2-matvec-and-H2-matmul)
* [Comparative Tests on KME, BKM, and VWF](https://github.com/huanghua1994/H2Pack/wiki/Comparative-Tests-on-KME-BKM-and-VWF)
## Last But Not Least

* [Can I ...](https://github.com/huanghua1994/H2Pack/wiki/Can-I)

