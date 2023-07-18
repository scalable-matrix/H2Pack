H2Pack is a library that provides linear-scaling storage and
linear-scaling matrix-vector multiplication for dense kernel matrices.
This is accomplished by storing the kernel matrices in the $\mathcal{H}^2$
hierarchical block low-rank representation.  It can be used for
Gaussian processes, solving integral equations, Brownian dynamics,
and other applications.

The main strength of H2Pack is its ability to efficiently construct $\mathcal{H}^2$ 
matrices in linear time for kernel functions used in Gaussian processes (up
to 3-D data) by using a new proxy point method.  Kernel functions from
computational physics, e.g., Coulomb, Stokes, can also be used.  H2Pack is
optimized for shared-memory multicore architectures, including the use
of vectorization for evaluating kernel functions.  H2Pack provides C/C++
and Python interfaces.

**Notes**

* H2Pack is designed for general kernel functions, including the Gaussian,
Matern, and other kernels used in statistics and machine learning.
For these kernels, H2Pack computes the $\mathcal{H}^2$ representation
much faster, and with linear complexity, compared to algebraic approaches
that rely on rank-revealing matrix decompositions.

* For standard kernel functions from potential theory, e.g., Coulomb, Stokes,
H2Pack using the proxy point method constructs a more efficient representation
than approaches based on analytic expansions, like the fast multipole method (FMM),
and thus has faster matrix-vector multiplication than FMM. Note that H2Pack requires
a preprocessing step to construct the $\mathcal{H}^2$
representation, while FMM does not need a preprocessing step.
However, FMM cannot handle general kernel functions.

* The proxy points only need to be computed once for a given kernel function, domain,
and requested accuracy. These proxy points can be reused for different sets
of points or data. Constructing the $\mathcal{H}^2$ matrix with these proxy points 
only requires linear time.
Alternatively, the proxy points could be selected on a surface, which
corresponds to the proxy surface method that can be useful
for kernel functions from potential theory.

**Other Features**
* Users can provide a function that defines the kernel function
or use kernels that are already built into H2Pack.
Vector wrapper functions are provided to help users optimize
the evaluation of their own kernel functions.
* HSS hierarchical block low-rank representations are also available,
including ULV decomposition and solve.
* A MATLAB version of H2Pack is available in [this repo](https://github.com/xinxing02/H2Pack-Matlab).

**Limitations**

* Kernel functions up to 3-dimensions
* Symmetric, translationally-invariant, non-oscillatory kernel functions
* H2Pack currently only supports kernel matrices defined by
a single set of points (i.e., square, symmetric matrices)

**Main Functions**

* $\mathcal{H}^2$ matrix representation construction for a kernel matrix with $O(N)$ complexity for an $N \times N$ matrix
* $\mathcal{H}^2$ matrix-vector multiplication with $O(N)$ complexity
* $\mathcal{H}^2$ matrix-matrix multiplication with $O(N)$ complexity

**References**

Please cite the following two papers if you use H2Pack in your work:

```bibtex
@article{xin2019,
    title = {Interpolative decomposition via proxy points for kernel matrices},
    journal = {SIAM Journal on Matrix Analysis and Applications},
    author = {Xing, Xin and Chow, Edmond},
    year = {2020},
    volume = {41},
    pages = {221--243},
}
```

```bibtex
@article{huang2020toms,
    title = { {H2Pack}: High-performance \textit{{H}} $^{\textrm{2}}$ Matrix Package for Kernel Matrices Using the Proxy Point Method },
    journal = {ACM Transactions on Mathematical Software},
    author = {Huang, Hua and Xing, Xin and Chow, Edmond},
    year = {2020},
    month = {Dec},
    volume = {47},
    pages = {1--29},
    doi = {10.1145/3412850},
    issn = {0098-3500, 1557-7295},
    number = {1},
}
```

H2Pack also implements other $\mathcal{H}^2$-related algorithms.

If you use the SPDHSS-H2 preconditioner (`H2P_SPDHSS_H2_build()`) in H2Pack, please cite the follow paper.

```bibtex
@article{xing2021,
    author = {Xing, Xin and Huang, Hua and Chow, Edmond},
    title = {Efficient Construction of an {HSS} Preconditioner for Symmetric Positive Definite \$\mathcal{H}^2\$ Matrices},
    journal = {SIAM Journal on Matrix Analysis and Applications},
    volume = {42},
    number = {2},
    pages = {683--707},
    year = {2021},
    doi = {10.1137/20M1365776},
}
```

If you use the periodic RPY kernel in H2Pack, please cite the follow paper.

```bibtex
@article{xing2022,
    title = {A Hierarchical Matrix Approach for Computing Hydrodynamic Interactions},
    author = {Xing, Xin and Huang, Hua and Chow, Edmond},
    journal = {Journal of Computational Physics},
    volume = {448},
    pages = {110761},
    year = {2022},
    issn = {0021--9991},
    doi = {10.1016/j.jcp.2021.110761},
    
}
```

If you use the data-driven $\mathcal{H}^2$ matrix construction algorithm (`H2P_build_with_sample_point()`), please cite the follow paper.

```bibtex
@article{cai2023,
    title = {Data-Driven Construction of Hierarchical Matrices With Nested Bases},
    author = {Cai, Difeng and Huang, Hua and Chow, Edmond and Xi, Yuanzhe},
    journal = {SIAM Journal on Scientific Computing},
    volume = {0},
    number = {0},
    pages = {S24--S50},
    year = {2023},
    doi = {10.1137/22M1500848},
}
```


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
* [H2 Matrix File Storage Scheme (draft)](https://github.com/scalable-matrix/H2Pack/wiki/H2-Matrix-File-Storage-Scheme)
* [Using H2 Matrix File Storage](https://github.com/scalable-matrix/H2Pack/wiki/Using-H2-Matrix-File-Storage)

## Numerical Tests

* [Accuracy Tests on Various Kernels](https://github.com/scalable-matrix/H2Pack/wiki/Accuracy-Tests-on-Various-Kernels)
* [Linear Scaling Tests](https://github.com/scalable-matrix/H2Pack/wiki/Linear-Scaling-Tests)
* [Parallel Efficiency Tests](https://github.com/scalable-matrix/H2Pack/wiki/Parallel-Efficiency-Tests)
* [Comparative Tests on H2-matvec and H2-matmul](https://github.com/scalable-matrix/H2Pack/wiki/Comparative-Tests-on-H2-matvec-and-H2-matmul)
* [Comparative Tests on KME, BKM, and VWF](https://github.com/scalable-matrix/H2Pack/wiki/Comparative-Tests-on-KME-BKM-and-VWF)

## Last But Not Least

* [Can I ...](https://github.com/scalable-matrix/H2Pack/wiki/Can-I)

