H2Pack is a high-performance, shared-memory library for constructing and operating with ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2) matrix representations for kernel matrices. In the current version, the operations available are:

* ![](https://latex.codecogs.com/svg.latex?O(N)) complexity ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)  matrix representation construction of a kernel matrix
* ![](https://latex.codecogs.com/svg.latex?O(N)) complexity ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)  matrix-vector multiplication
* ![](https://latex.codecogs.com/svg.latex?O(N)) complexity ![](https://latex.codecogs.com/svg.latex?\mathcal{H}^2)  matrix - dense matrix (multiple vectors) multiplication

H2Pack is written in C99 and provides a C/C++ interface. H2Pack also has an experimental Python interface. 

Please cite the following papers if you use H2Pack in your work:

* [Interpolative Decomposition via Proxy Points For Kernel Matrices](https://www.cc.gatech.edu/~echow/pubs/xing-chow-simax-2019.pdf)
* [H2Pack: High-Performance H2 Matrix Package for Kernel Matrices Using the Proxy Point Method]()



## Getting Started

* [Installing H2Pack](https://github.com/huanghua1994/H2Pack/wiki/Installing-H2Pack)
* [Using H2Pack in C/C++](https://github.com/huanghua1994/H2Pack/wiki/Using-H2Pack-in-C-CPP)
*  [Using and Writing Kernel Functions](https://github.com/huanghua1994/H2Pack/wiki/Using-and-Writing-Kernel-Functions) 
* [Using H2Pack in Python](https://github.com/huanghua1994/H2Pack/wiki/Using-H2Pack-in-Python)


## Using H2Pack Efficiently

* [Choosing the Running Mode (AOT/JIT)](https://github.com/huanghua1994/H2Pack/wiki/Choosing-the-Running-Mode)
* [Writing Bi-Kernel Matvec (BKM) Functions](https://github.com/huanghua1994/H2Pack/wiki/Writing-BKM-Functions)
* [Using Vector Wrapper Functions in Kernel Functions](https://github.com/huanghua1994/H2Pack/wiki/Using-VWF-in-Kernel-Functions)

## Benchmarks

* [Benchmarks On a Desktop](https://github.com/huanghua1994/H2Pack/wiki/Benchmarks-on-A-Desktop)
* [Benchmarks On a Server](https://github.com/huanghua1994/H2Pack/wiki/Benchmarks-on-A-Server)

## Last But Not Least

* [Can I ...](https://github.com/huanghua1994/H2Pack/wiki/Can-I)