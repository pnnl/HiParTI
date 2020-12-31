HiParTI
------

A Hierarchical Parallel Tensor Infrastructure (HiParTI), is to support fast essential sparse tensor operations and tensor decompositions on multicore CPU and GPU architectures. These basic tensor operations are critical to the overall performance of tensor analysis algorithms (such as tensor decomposition). HiParTI is based on [[ParTI!]](https://github.com/hpcgarage/ParTI) library developed at GaTech. 


# Contents:

## Supported Data
* General sparse tensors/matrices
* Semi-sparse tensors/matrices with dense dimensions

## Sparse tensor representations:
* Coordinate (COO) format
* Hierarchical Coordinate (HiCOO) format (Refer to [[SC'19 paper]](http://fruitfly1026.github.io/static/files/sc18-li.pdf))
* Semi-COO (sCOO) format (Refer to [[pdf]](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf))
* Semi-HiCOO (sHiCOO) format (Refer to [[pdf]](http://fruitfly1026.github.io/static/files/iiswc20-li.pdf))

## Sparse tensor operations:

* Scala-tensor mul/div
* Element-wise tensor add/sub/mul/div
* Sparse tensor-times-dense vector (SpTTV)
* Sparse tensor-times-dense matrix (SpTTM)
* Sparse matricized tensor times Khatri-Rao product (SpMTTKRP)
* Sparse tensor contraction (SpTC)
* Sparse tensor sorting
* Sparse tensor reordering
* Sparse tensor matricization
* Kronecker product
* Khatri-Rao product

## Sparse tensor decompositions:

* Sparse CANDECOMP/PARAFAC decomposition
* Sparse Tucker decomposition

# Build requirements:

- C Compiler (GCC or Clang)

- [CUDA SDK](https://developer.nvidia.com/cuda-downloads) (Optional)

- [CMake](https://cmake.org) (>v3.0)

- BLAS: E.g., [OpenBLAS](http://www.openblas.net), Intel MKL, or [MAGMA](http://icl.cs.utk.edu/magma/)


# Build:

<!-- 1. Create a file by `touch build.config' to define OpenBLAS_DIR and MAGMA_DIR -->
1. Create a file by `cp build-sample.config build.config` to open and/or define library path (Please leave -DUSE_OPENMP=ON for now. Also please explicitly define compliers using `-DCMAKE_C_COMPILER` and `-DCMAKE_CXX_COMPILER`)

2. Type `./build.sh`

3. Check `build` for resulting library

4. Check `build/benchmark` for example programs


# Build docs:

1. Install Doxygen

2. Go to `docs`

3. Type `make`


# Build MATLAB interface:

1. `cd matlab`

2. export LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH

3. Type `make` to build all functions into MEX library.

4. matlab

    1. In matlab environment, type `addpath(pwd)`
   
    2. Play with ParTI MATLAB inferface.
    

<br/>The algorithms and details are described in the following publications.
# Publications
* **Sparta: High-Performance, Element-Wise Sparse Tensor Contraction on Heterogeneous Memory**. Jiawen Liu, Jie Ren, Roberto Gioiosa, Dong Li, Jiajia Li. Principles and Practice of Parallel Programming (PPoPP). 2021. (Accepted)
* **Generic, Sparse Tensor Core for Neural Networks**. Xiaolong Wu, Yang Yi, Dave (Jing) Tian, Jiajia Li. 1st International Workshop on Machine Learning for Software Hardware Co-Design (MLSH), in conjunction with the 29th International Conference on Parallel Architectures and Compilation Techniques (PACT). 2020
* **Programming Strategies for Irregular Algorithms on the Emu Chick**. Eric Hein, Srinivas Eswar, Abdurrahman Yasar, Jiajia Li, Jeffrey S. Young, Tom Conte, Umit V. Catalyurek, Rich Vuduc, Jason Riedy, Bora Ucar. Transactions on Parallel Computing. 2020. (Accepted)
* **Sparsity-Aware Distributed Tensor Decomposition**. Zheng Miao, Jon C. Calhoun, Rong Ge, Jiajia Li. ACM/IEEE International Conference for High-Performance Computing, Networking, Storage, and Analysis (SC). 2020. (Poster)
* **Efficient and Effective Sparse Tensor Reordering**. Jiajia Li, Bora Ucar, Umit Catalyurek, Kevin Barker, Richard Vuduc. International Conference on Supercomputing (ICS). 2019.
* **A Microbenchmark Characterization of the Emu Chick**. Jeffrey S. Young, Eric Hein, Srinivas Eswar, Patrick Lavin, Jiajia Li, Jason Riedy, Richard Vuduc, Thomas M. Conte. Journal of Parallel Computing. 2019.
* **HiCOO: Hierarchical Storage of Sparse Tensors**. Jiajia Li, Jimeng Sun, Richard Vuduc. ACM/IEEE International Conference for High-Performance Computing, Networking, Storage, and Analysis (SC). 2018. (Best Student Paper Award) [[pdf]](http://fruitfly1026.github.io/static/files/sc18-li.pdf)
* **Optimizing Sparse Tensor Times Matrix on GPUs**. Yuchen Ma, Jiajia Li, Xiaolong Wu, Chenggang Yan, Jimeng Sun, Richard Vuduc. Journal of Parallel and Distributed Computing (Special Issue on Systems for Learning, Inferencing, and Discovering). 2018.
* **Optimizing Sparse Tensor Times Matrix on multi-core and many-core architectures**. Jiajia Li, Yuchen Ma, Chenggang Yan, Richard Vuduc. The sixth Workshop on Irregular Applications: Architectures and Algorithms (IA^3), co-located with SC’16. 2016. [[pdf]](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf)
* **ParTI!: a Parallel Tensor Infrastructure for Data Analysis**. Jiajia Li, Yuchen Ma, Chenggang Yan, Jimeng Sun, Richard Vuduc. Tensor-Learn Workshop @ NIPS'16. [[pdf]](http://fruitfly1026.github.io/static/files/nips16-tensorlearn.pdf)


# Contributors

* Jiajia Li (Contact: Jiajia.Li@pnnl.gov or fruitfly1026@gmail.com)
