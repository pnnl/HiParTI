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
3. `git clone https://gitlab.com/jiawenliu64/tensors` 
Contains 1. Tensor datasets; 2. The raw data of all experiments in the paper (named "Collected_Results.xlsx") for comparison.
(The three software above is also provided in the "Artifact download URL" in the PPoPP AE submission)

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
    
    
# Step-by-Step Instructions for Results Reproduction:

## Remote Access to Our Machines:

Our experiments are profiled on two machines: One multi-core CPU machine in Chameleon Cloud and one Intel Optane machine in UC Merced's server room. **If reviewers would like to use our machines, please contact us to reserve the time slot to avoid experiment conflict with other reviewers.**

### Intel Optane DC PMM Server
1. Configuration: Intel Xeon Cascade-Lake CPU including 24 physical cores with 2.3 GHz frequency. Each socket has 6 × 16 GB of DRAM and 6 × 128 GB Intel Optane DIMMs.
2. Login instruction: 
* Use `ssh ae@169.236.180.218` to log in the Intel Optane server with password: `ppopp`. 
* Reviewer 2 and Reviewer 3 can use the account `ae2` and `ae3` correspondingly with the same password `ppopp`.


### Chameleon Cloud
1. Configuration: 96 GB DDR4 memory, Intel Xeon Gold 6126 CPU including 12 physical cores with 2.6 GHz frequency on one socket 
2. Login instruction: 
* `ssh ae@192.5.87.44`
* Password: `ppopp`
* Reviewer 2 and Reviewer 3 can use the account `ae2` and `ae3` correspondingly with the same password `ppopp`.

## Reproduce Results on an Intel Optane DC PMM Server:

Use the following steps to reproduce experiments in the paper. Note that the following experiments must be executed on an server with Intel Optane DC PMM. For your convenience, you can access our machine in the `Remote Access to Our Machines` section. The default setting of our machine is the `App Direct NUMA Mode`. The modes configuration instruction is also listed below.

You might need the following steps to switch to `Memory Mode` only in the Intel Optane Server (not for the General Linux Server) for some experiments listed as `Memory Mode`.

1. Switch to `Memory Mode` (for "Memory Mode" in Figures 7, 8, 9): 
* `sudo path/to/sparta/optane_setup/memory_mode.sh` (A reboot will be automatically performed and users need to login manually.)

2. Switch to `App Direct NUMA Mode` (for "Sparta, IAL, Optane-only and DRAM-only" in Figures 7, 8, 9): 
* `sudo sparta/optane_setup/numa_mode1.sh` (A reboot will be automatically performed and users need to login manually) & (after reroot)`sudo sparta/optane_setup/numa_mode2.sh` 

Please use the following three steps to reproduce the experiments, which can be run separately as below (i.e., Figures 3, 7, 8 and 9).

1. Sparta on Heterogeneous Memory Systems: Performance (i.e., Figures 7) including Execution Time Breakdown (i.e., Figure 3): 
* `cd sparta/run_optane` & `./run_speedup.sh > run_speedup_result.txt` . After running this script, the results are stored in the `run_speedup_result.txt` file. To convert the results from .txt format into the .csv format, please run `python3 path/to/sparta/output_scripts/run_speedup_optane_result.py run_speedup_optane_result.txt` for 4 execution modes (i.e., Sparta, IAL, Optane-only and DRAM-only).
(When executing IAL on our machine, users may be asked once for entering the password, which is `ppopp`.) 
* `cd sparta/run_optane` & `./run_speedup_memory_mode.sh > run_speedup_memory_mode_result.txt`. After running this script, the results are stored in the `run_speedup_memory_mode_result.txt` file. To convert the results from .txt format into the .csv format, please run `python3 path/to/sparta/output_scripts/run_speedup_memory_mode_result.py run_speedup_optane_memory_mode_result.txt` for Memory Mode. (Users need to switch to Intel Optane `Memory Mode` manually using the instructions above and then execute this command.)
* If users want to also profile the memory bandwidth (i.e., Figure 8), please combine step 2 with this step.
* If users want to also profile the peak memory consumption (i.e., Figure 9), please combine step 3 with this step.

Note that this experiment is very time-consuming. It took us around 7 hours to complete all experiments. 

(When performing a long-running task (e.g., Chicago* with 2-Mode on Optane-only case) on a remote machine, your network connection might suddenly lose. You can use `screen` command to execute the above scripts. For example, you can run `screen -R run`, execute the scripts, record the `screen_id`, run `control + A + D` to detach screen, and run `screen -R run screen_id` to jump into the long-running task execution.)

2. Sparta on Heterogeneous Memory Systems: Bandwidth (i.e., Figure 8): 
* `cd sparta/run_optane` & run `./run_bw.sh` in one terminal & run `./run_speedup.sh` in another terminal. The execution script already generates a `bw.csv` file. The sum of columns "DRAMRead" and "DRAMWrite" is corresponding to "DRAM" in Figure 8 (the upper one); The sum of columns "PMMREAD" and "PMMWrite" is corresponding to "PMM" in Figure 8 (the bottom one).
(Users are asked once for entering the password, which is `ppopp`.)

3. Sparta on Heterogeneous Memory Systems: Peak Memory Consumption (i.e., Figure 9): 
* `cd sparta/run_optane` & run `./run_memory.sh` in one terminal & run `./run_speedup.sh` in another terminal. The maximum value under the `used` column in the generated file `mem.txt` is the corresponding value in Figure 9.


We also support running all experiments (Figures 2-9) on one Intel Optane server. You can run the following script: 
* `cd sparta` & `./run.sh`

However, the results on DRAM memory will be slightly different from Figures 4, 2, 5 and 6 in the paper, due to the different CPU core frequencies. The results in the paper were collected on the Chameleon Cloud server, due to the high competition among multiple users of the Intel Optane DC PMM server.

## Experiments on a General Linux Server:
To reproduce the exact results of Figures 4, 2, 5 and 6 as shown in the paper, a general Linux server with multicore CPUs is enough. For your convenience, you can access our Chameleon Cloud machine in the `Remote Access to Our Machines` section. Please use the following steps to reproduce the experiments in "Overall Performance" (including Execution Time Breakdown), "Performance Comparison to ITensor" and "Thread Scalability" (i.e., Figures 2, 4, 5 and 6)

1. Overall Performance (i.e., Figures 4 and 2): 
* `cd sparta/run` & `./run_speedup.sh > run_speedup_result.txt`. After running this script, the results are stored in the `run_speedup_result.txt` file. To convert the results from .txt format into the .csv format, please run `python3 path/to/sparta/output_scripts/run_speedup_result.py run_speedup_result.txt`.

2. Performance Comparison to ITensor (i.e., Figure 5): 
* `cd sparta/run` & `./run_itensor.sh > run_itensor_result.txt`. After running this script, the results are stored in the `run_itensor_result.txt` file. To convert the results from .txt format into the .csv format, please run `python3 path/to/sparta/output_scripts/run_itensor_result.py run_itensor_result.txt`. 

3. Thread Scalability (i.e., Figure 6): 
* `cd sparta/run` & `./run_scalability.sh > run_scalability_result.txt`. After running this script, the results are stored in the `run_scalability_result.txt` file. To convert the results from .txt format into the .csv format, please run `python3 path/to/sparta/output_scripts/run_scalability_result.py run_scalability_result.txt`.


# More Support

## Tensor Contraction Parameters:
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

## ITensor Results Generation:

We have generated the sparse tensors and performance from ITensor library and stored them in `itensor/results`. If you want to recollect all these tensors, you can use the following steps.

1. `git clone https://gitlab.com/jiawenliu64/itensor` (forked from [ITensor repo](https://github.com/ITensor/ITensor), also provided in the "Artifact download URL" in the PPoPP AE submission.)
2. `export ITENSOR_DIR=path/to/itensor`
3. `mkdir path/to/itensor_results` & `export ITENSOR_RESULTS=path/to/itensor_results`
4. `cd $ITENSOR_DIR` & `run.sh`
5. `cd $ITENSOR_DIR/hubbard` & `OMP_NUM_THREADS=12 ./main 'parity' 1`. After the execution, all results are stored in `$ITENSOR_RESULTS`. The result (execution time) is included in the second line of each generated file (e.g., tensor_2137.txt).

* If you also want to convert the data to the .bin format as they are shown in `path/to/itensor/results`, you can use the following steps to process data using SPLATT, another sparse tensor library. 

1. `git clone https://github.com/ShadenSmith/splatt`
2. `./configure --prefix=SPLATT_DIR` & `make` & `make install`
3. Replace all `Block` in tensor A to `A-Block`. For example, in vim, you can execute `x,ys/Block/A-Block/g` to replace from line `x` to line `y`.
4. Replace all `Block` in tensor B to `B-Block`. For example, in vim, you can execute `x,ys/Block/B-Block/g` to replace from line `x` to line `y`.
5. `python path/to/sparta/output_scripts/gen_tns_itensor.py path/to/itensor_results/tensor_x.txt 0.00000001` for data tensor_x.
6. `path/to/splatt/build/Linux-x86_64/bin/splatt convert -t bin path/to/itensor_results/tensor_x_A.tns path/to/itensor_results/tensor_x_A.bin` for A in x.
7. `path/to/splatt/build/Linux-x86_64/bin/splatt convert -t bin path/to/itensor_results/tensor_x_B.tns path/to/itensor_results/tensor_x_B.bin` for B in x.

# Contributiors

* Jiawen Liu (Contact: jliu265@ucmerced.edu)
* Jiajia Li (Contact: Jiajia.Li@pnnl.gov)

