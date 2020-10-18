H2Pack is a high-performance, shared-memory library for **linear-scaling matrix-vector multiplication of kernel matrices** based on their ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2) matrix representations. 
The key feature of H2Pack is the efficient ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2) matrix construction for kernel matrices using a hybrid compression method called the proxy point method. 

H2Pack supports kernel matrices of the form _K(X,X)_ defined by a translationally-invariant, symmetric kernel function _K(x,y)_ (e.g., Gaussian, Matern, Laplace, and Stokes kernels) and a point set _X_ in low-dimensional space (i.e., 1D, 2D, and 3D). You can provide a function that defines the kernel or use
kernels that are already built into H2Pack.
H2Pack, written in C99, provides a C/C++ interface and an experimental Python interface.
<!-- The Matlab prototype of H2Pack can be found in the [repo](). -->

**Main Functions:**

* _O(N)_ complexity ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)  matrix representation construction for a kernel matrix (![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)-construction)
* _O(N)_ complexity ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)  matrix-vector multiplication (![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)-matvec)
* _O(N)_ complexity ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)  matrix-matrix  multiplication (![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)-matmul)


**References**:

* X. Xing and E. Chow, Interpolative decomposition via proxy points for kernel matrices, _SIAM Journal on Matrix Analysis and Applications_, 41(1), 221–243 (2020)
* H. Huang, X. Xing, and E. Chow, H2Pack: High-performance H2 matrix package for kernel matrices using the proxy point method, _ACM Transactions on Mathematical Software_, to appear (2020)

**Additional Features:** 

H2Pack also provides functions for constructing and applying HSS matrix representations: 

* HSS matrix representation construction for a kernel matrix using the proxy point method 
* HSS matrix-vector multiplication
* HSS matrix-matrix multiplication
* ULV decomposition of HSS matrix representation
* Direct solves involving the HSS matrix representation using its ULV decomposition


## Getting Started

* [Installing H2Pack](https://github.com/scalable-matrix/H2Pack/wiki/Installing-H2Pack)
* [Basic Usage](https://github.com/scalable-matrix/H2Pack/wiki/Basic-Usage)
* [Using and Writing Kernel Functions](https://github.com/scalable-matrix/H2Pack/wiki/Using-and-Writing-Kernel-Functions) 
* [Two Running Modes for H2Pack](https://github.com/scalable-matrix/H2Pack/wiki/Two-Running-Modes-for-H2Pack)
* [HSS-Related Computations](https://github.com/scalable-matrix/H2Pack/wiki/HSS-Related-Computations)

## Advanced Configurations and Tools

* [Bi-Kernel Matvec (BKM) Functions](https://github.com/scalable-matrix/H2Pack/wiki/Bi-Kernel-Matvec-Functions)
* [Vector Wrapper Functions for Kernel Evaluations](https://github.com/scalable-matrix/H2Pack/wiki/Vector-Wrapper-Functions-For-Kernel-Evaluations)
* [Proxy Points and their Reuse](https://github.com/scalable-matrix/H2Pack/wiki/Proxy-Points-and-their-Reuse)
* [Python Interface](https://github.com/scalable-matrix/H2Pack/wiki/Using-H2Pack-in-Python)


## Numerical Tests

* [Accuracy Tests on Various Kernels](https://github.com/scalable-matrix/H2Pack/wiki/Accuracy-Tests-on-Various-Kernels)
* [Linear Scaling Tests](https://github.com/scalable-matrix/H2Pack/wiki/Linear-Scaling-Tests)
* [Parallel Efficiency Tests](https://github.com/scalable-matrix/H2Pack/wiki/Parallel-Efficiency-Tests)
* [Comparative Tests on H2-matvec and H2-matmul](https://github.com/scalable-matrix/H2Pack/wiki/Comparative-Tests-on-H2-matvec-and-H2-matmul)
* [Comparative Tests on KME, BKM, and VWF](https://github.com/scalable-matrix/H2Pack/wiki/Comparative-Tests-on-KME-BKM-and-VWF)

## Last But Not Least

* [Can I ...](https://github.com/scalable-matrix/H2Pack/wiki/Can-I)

