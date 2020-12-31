Sparta
-----------

Sparta is a high-performance, element-wise sparse tensor contraction software on multicore CPU with heterogeneous memory. The sparse tensor contraction is critical to the overall performance of scientific applications, such as quantum chemistry and quantum physics. Sparta is implemented based on the open-sourced Hierarchical Parallel Tensor Infrastructure (HiParTI) library (https://gitlab.com/tensorworld/hiparti), which supports fast essential sparse tensor operations with matrices and vectors and tensor decompositions on multicore CPU and GPU architectures. 

# Build requirements:

- [GNU Compiler (GCC)](https://gcc.gnu.org/) (>=v7.5)
- [CMake](https://cmake.org) (>=v3.0)
- [OpenBLAS](http://www.openblas.net)
- [NUMA](https://linux.die.net/man/3/numa)

You may use the following steps to install the required libraries:

* OpenBLAS
1. `git clone https://github.com/xianyi/OpenBLAS`
2. `cd OpenBLAS`
3. `make -j`
4. `mkdir path/to/OpenBLAS_install`
5. `make install PREFIX=path/to/OpenBLAS_install`
6. Append `export OpenBLAS_DIR=path/to/OpenBLAS_install` to `~/.bashrc`

* CMake
1. `sudo apt-get install cmake`

* NUMA
1. `sudo apt-get install libnuma-dev`
2. `sudo apt-get install numactl`


# Download & Set up Projects:

## Download
1. `git clone https://gitlab.com/jiawenliu64/sparta` (Sparta)
2. `git clone https://gitlab.com/jiawenliu64/ial` (IAL)
3. `git clone https://gitlab.com/jiawenliu64/tensors` (Datasets)

## Build
* `cd sparta` & `./build.sh`

## Set the path environments:
You can execute the commands, e.g., `export SPARTA_DIR=path/to/sparta`, prior to the execution or append these commands to `~/.bashrc`.

* `SPARTA_DIR` (path/to/sparta, e.g., /home/ae/sparta)
* `IAL_SRC` (path/to/IAL, e.g., /home/ae/ial/src)
* `TENSOR_DIR` (path/to/tensors, /home/ae/tensors)

You can execute a command like "export EXPERIMENT_MODES=x" to set up the environment variable for different test purposes. (This step has been included in our scripts below, so you don't need to explicitly specify it.)
* `export EXPERIMENT_MODES=0`: COOY + SPA
* `export EXPERIMENT_MODES=1`: COOY + HtA
* `export EXPERIMENT_MODES=3`: HtY + HtA
* `export EXPERIMENT_MODES=4`: HtY + HtA on Optane

## A Test Run

1. On a general multicore CPU server with Linux:
* `./sparta/run/test_run.sh`

2. On a server with Intel Optane DC PMM:
* `./sparta/run_optane/test_run.sh`
    
# Tensor Contraction Parameters:
You can check the parameters options with `path/to/sparta/build/ttt --help`
```
Options: -X FIRST INPUT TENSOR
         -Y FIRST INPUT TENSOR
         -Z OUTPUT TENSOR (Optinal)
         -m NUMBER OF CONTRACT MODES
         -x CONTRACT MODES FOR TENSOR X (0-based)
         -y CONTRACT MODES FOR TENSOR Y (0-based)
         -t NTHREADS, --nt=NT (Optinal)
         --help
```

# Contributiors

* Jiawen Liu (Contact: jliu265@ucmerced.edu)
* Jiajia Li (Contact: Jiajia.Li@pnnl.gov)

