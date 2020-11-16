H2Pack is a library that provides linear-scaling storage and
linear-scaling matrix-vector multiplication for dense kernel matrices.
This is accomplished by storing the kernel matrices in the
![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2) or HSS
hierarchical block low-rank representations.  Applications include
integral equations, Gaussian processes, Brownian dynamics, and others.

**Features**

* H2Pack is designed for general kernel functions, including the Gaussian,
Matern, and other kernels used in statistics and machine learning. This is
due to the use of a new proxy point method used to construct the matrix
representations.  The common proxy surface method is also provided to
efficiently construct matrix representations for kernels from potential
theory, i.e., Coulomb, Stokes, etc.

* H2Pack requires less storage (for the same accuracy) than *analytic*
methods such as the fast multipole method (FMM).  H2Pack is faster
than *algebraic* methods such as those that rely on rank-revealing matrix
decompositions.  This is due to the hybrid analytic-algebraic approach
of the proxy point method.

* H2Pack achieves high-performance on shared-memory multicore
architectures by using multithreading, vectorization, and careful load
balancing.  Users can provide a function that defines the kernel function
or use kernels that are already built into H2Pack.
Vector wrapper functions are provided to help users optimize
the evaluation of their own kernel functions.

* H2Pack provides both C/C++ and Python interfaces.
A Matlab version of H2Pack is also available in this repo.

**Limitations**

* Kernel functions up to 3-dimensions
* Non-oscillatory kernel functions
* Translationally-invariant kernel functions
* Symmetric kernel functions
* H2Pack currently only supports kernel matrices defined by
a single set of points (i.e., square, symmetric matrices)

**Main Functions**

* ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2) matrix representation construction for a kernel matrix (![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)-construction) with _O(N)_ complexity
* ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2) matrix-vector multiplication (![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)-matvec) with _O(N)_ complexity
* ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2) matrix-matrix  multiplication (![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)-matmul) with _O(N)_ complexity
* HSS matrix representation construction for a kernel matrix using the proxy point method 
* HSS matrix-vector multiplication
* HSS matrix-matrix multiplication
* ULV decomposition of HSS matrix representation
* Direct solves involving the HSS matrix representation using its ULV decomposition

**References**

* H. Huang, X. Xing, and E. Chow, [H2Pack: High-performance H2 matrix package for kernel matrices using the proxy point method](https://www.cc.gatech.edu/~echow/pubs/h2pack.pdf), _ACM Transactions on Mathematical Software_, to appear (2020).
* X. Xing and E. Chow, [Interpolative decomposition via proxy points for kernel matrices](https://www.cc.gatech.edu/~echow/pubs/xing-chow-simax-2019.pdf), _SIAM Journal on Matrix Analysis and Applications_, 41(1), 221–243 (2020).


## Getting Started

* [Installing H2Pack](https://github.com/scalable-matrix/H2Pack/wiki/Installing-H2Pack)
* [Basic Application Interface](https://github.com/scalable-matrix/H2Pack/wiki/Basic-Usage)
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

