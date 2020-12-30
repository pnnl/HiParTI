/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <omp.h>
#include <HiParTI.h>
#include "../src/sptensor/sptensor.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -m MODE, --mode=MODE (default -1: loop all modes)\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -s sortcase, --sortcase=SORTCASE (1,2,3,4)\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t NTHREADS, --nt=NT\n");
    printf("         -u use_reduce, --ur=use_reduce\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    char ifname[1000];
    FILE *fo = NULL;
    ptiSparseTensor X;
    ptiMatrix ** U;
    ptiMatrix ** copy_U;

    ptiIndex mode = HIPARTI_INDEX_MAX;
    ptiIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads = 1;
    int impl_num = 0;
    int use_reduce = 1; // Need to choose from two omp parallel approaches
    int nt = 1;
    /* sortcase:
     * = 0 : the same with the old COO code.
     * = 1 : best case. Sort order: [mode, (ordered by increasing dimension sizes)]
     * = 2 : worse case. Sort order: [(ordered by decreasing dimension sizes)]
     * = 3 : Z-Morton ordering (same with HiCOO format order)
     * = 4 : random shuffling.
     */
    int sortcase = 0;
    ptiElementIndex sb_bits;
    ptiElementIndex sk_bits;
    printf("niters: %d\n", niters);

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},//input file
            {"mode", required_argument, 0, 'm'},//
            {"output", optional_argument, 0, 'o'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"sortcase", optional_argument, 0, 's'},
            {"impl-num", optional_argument, 0, 'p'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"nt", optional_argument, 0, 't'},
            {"use-reduce", optional_argument, 0, 'u'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:b:k:s:p:d:r:t:u:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            strcpy(ifname, optarg);
            printf("input file: %s\n", ifname); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "aw");
            ptiAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'm':
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &mode);
            break;
        case 'b':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 's':
            sscanf(optarg, "%d", &sortcase);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%u"HIPARTI_SCN_INDEX, &R);
            break;
        case 'u':
            sscanf(optarg, "%d", &use_reduce);
            break;
        case 't':
            sscanf(optarg, "%d", &nt);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }

    printf("mode: %"HIPARTI_PRI_INDEX "\n", mode);
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    printf("sortcase: %d\n", sortcase);

    /* Load a sparse tensor from file as it is */
    ptiAssert(ptiLoadSparseTensor(&X, 1, ifname) == 0);
    ptiSparseTensorStatus(&X, stdout);
    // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);

    ptiIndex nmodes = X.nmodes;
    U = (ptiMatrix **)malloc((nmodes+1) * sizeof(ptiMatrix*));
    for(ptiIndex m=0; m<nmodes+1; ++m) {
      U[m] = (ptiMatrix *)malloc(sizeof(ptiMatrix));
    }
    ptiIndex max_ndims = 0;
    for(ptiIndex m=0; m<nmodes; ++m) {
      ptiAssert(ptiNewMatrix(U[m], X.ndims[m], R) == 0);
      ptiAssert(ptiConstantMatrix(U[m], 1) == 0);
      // ptiAssert(ptiRandomizeMatrix(U[m]) == 0);
      if(X.ndims[m] > max_ndims)
        max_ndims = X.ndims[m];
    }
    ptiAssert(ptiNewMatrix(U[nmodes], max_ndims, R) == 0);
    ptiAssert(ptiConstantMatrix(U[nmodes], 0) == 0);
    ptiIndex stride = U[0]->stride;

    ptiIndex * mode_order = (ptiIndex*) malloc(X.nmodes * sizeof(*mode_order));
    ptiIndex * mats_order = (ptiIndex*)malloc(nmodes * sizeof(ptiIndex));

    /* Initialize locks */
    ptiMutexPool * lock_pool = NULL;
    if(cuda_dev_id == -1 && use_reduce == 0) {
        lock_pool = ptiMutexAlloc();
    }


    if (mode == HIPARTI_INDEX_MAX) {

        for(ptiIndex mode=0; mode<nmodes; ++mode) {
            /* Reset U[nmodes] */
            U[nmodes]->nrows = X.ndims[mode];
            ptiAssert(ptiConstantMatrix(U[nmodes], 0) == 0);

            /* Sort sparse tensor */
            memset(mode_order, 0, X.nmodes * sizeof(*mode_order));
            switch (sortcase) {
                case 0:
                    ptiSparseTensorSortIndex(&X, 1, nt);
                    break;
                case 1:
                    ptiGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
                    ptiSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nt);
                    break;
                case 2:
                    ptiGetWorstModeOrder(mode_order, mode, X.ndims, X.nmodes);
                    ptiSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nt);
                    break;
                case 3:
                    /* Pre-process tensor, the same with the one used in HiCOO.
                     * Only difference is not setting kptr and kschr in this function.
                     */
                    ptiSparseTensorMixedOrder(&X, sb_bits, sk_bits, nt);
                    break;
                case 4:
                    // ptiGetBestModeOrder(mode_order, 0, X.ndims, X.nmodes);
                    ptiGetRandomShuffleElements(&X);
                    break;
                default:
                    printf("Wrong sortcase number, reset by -s. \n");
            }
            if(sortcase != 0) {
                printf("mode_order:\n");
                ptiDumpIndexArray(mode_order, X.nmodes, stdout);
            }

            ptiSparseTensorStatus(&X, stdout);
            // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);


            /* Set zeros for temporary copy_U, for mode-"mode" */
            char * bytestr;
            if(cuda_dev_id == -1 && use_reduce == 1) {
                copy_U = (ptiMatrix **)malloc(nt * sizeof(ptiMatrix*));
                for(int t=0; t<nt; ++t) {
                    copy_U[t] = (ptiMatrix *)malloc(sizeof(ptiMatrix));
                    ptiAssert(ptiNewMatrix(copy_U[t], X.ndims[mode], R) == 0);
                    ptiAssert(ptiConstantMatrix(copy_U[t], 0) == 0);
                }
                ptiNnzIndex bytes = nt * X.ndims[mode] * R * sizeof(ptiValue);
                bytestr = ptiBytesString(bytes);
                printf("MODE MATRIX COPY=%s\n", bytestr);
                free(bytestr);
            }

            switch (sortcase) {
            case 0:
            case 3:
            case 4:
                mats_order[0] = mode;
                for(ptiIndex i=1; i<nmodes; ++i)
                    mats_order[i] = (mode+i) % nmodes;
                break;
            case 1: // Reverse of mode_order except the 1st one
                mats_order[0] = mode;
                for(ptiIndex i=1; i<nmodes; ++i)
                    mats_order[i] = mode_order[nmodes - i];
                break;
            case 2: // Totally reverse of mode_order
                for(ptiIndex i=0; i<nmodes; ++i)
                    mats_order[i] = mode_order[nmodes - i];
                break;
            }
            // printf("mats_order:\n");
            // ptiDumpIndexArray(mats_order, nmodes, stdout);


            /* For warm-up caches, timing not included */
            if(cuda_dev_id == -2) {
                nthreads = 1;
                ptiAssert(ptiMTTKRP(&X, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                printf("nt: %d\n", nt);
                if(use_reduce == 1) {
                    printf("ptiOmpMTTKRP_Reduce:\n");
                    ptiAssert(ptiOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
                } else {
                    printf("ptiOmpMTTKRP:\n");
                    ptiAssert(ptiOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
                    // printf("ptiOmpMTTKRP_Lock:\n");
                    // ptiAssert(ptiOmpMTTKRP_Lock(&X, U, mats_order, mode, nt, lock_pool) == 0);
                }
            }

            
            ptiTimer timer;
            ptiNewTimer(&timer, 0);
            ptiStartTimer(timer);

            for(int it=0; it<niters; ++it) {
                // ptiAssert(ptiConstantMatrix(U[nmodes], 0) == 0);
                if(cuda_dev_id == -2) {
                    nthreads = 1;
                    ptiAssert(ptiMTTKRP(&X, U, mats_order, mode) == 0);
                } else if(cuda_dev_id == -1) {
                    if(use_reduce == 1) {
                        ptiAssert(ptiOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
                    } else {
                        ptiAssert(ptiOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
                        // printf("ptiOmpMTTKRP_Lock:\n");
                        // ptiAssert(ptiOmpMTTKRP_Lock(&X, U, mats_order, mode, nt, lock_pool) == 0);
                    }
                }
            }

            ptiStopTimer(timer);

            if(cuda_dev_id == -2 || cuda_dev_id == -1) {
                char * prg_name;
                asprintf(&prg_name, "CPU  SpTns MTTKRP MODE %"HIPARTI_PRI_INDEX, mode);
                double aver_time = ptiPrintAverageElapsedTime(timer, niters, prg_name);

                double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
                uint64_t bytes = ( nmodes * sizeof(ptiIndex) + sizeof(ptiValue) ) * X.nnz;
                for (ptiIndex m=0; m<nmodes; ++m) {
                    bytes += X.ndims[m] * R * sizeof(ptiValue);
                }
                double gbw = (double)bytes / aver_time / 1e9;
                printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
            }
            ptiFreeTimer(timer);

            if(fo != NULL) {
                ptiAssert(ptiDumpMatrix(U[nmodes], fo) == 0);
            }

        } // End nmodes

    } else {
        /* Sort sparse tensor */
        memset(mode_order, 0, X.nmodes * sizeof(*mode_order));
        switch (sortcase) {
            case 0:
                ptiSparseTensorSortIndex(&X, 1, nt);
                break;
            case 1:
                ptiGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
                ptiSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nt);
                break;
            case 2:
                ptiGetWorstModeOrder(mode_order, mode, X.ndims, X.nmodes);
                ptiSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nt);
                break;
            case 3:
                /* Pre-process tensor, the same with the one used in HiCOO.
                 * Only difference is not setting kptr and kschr in this function.
                 */
                ptiSparseTensorMixedOrder(&X, sb_bits, sk_bits, nt);
                break;
            case 4:
                // ptiGetBestModeOrder(mode_order, 0, X.ndims, X.nmodes);
                ptiGetRandomShuffleElements(&X);
                break;
            default:
                printf("Wrong sortcase number, reset by -s. \n");
        }
        if(sortcase != 0) {
            printf("mode_order:\n");
            ptiDumpIndexArray(mode_order, X.nmodes, stdout);
        }

        ptiSparseTensorStatus(&X, stdout);
        // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);

        /* Set zeros for temporary copy_U, for mode-"mode" */
        char * bytestr;
        if(cuda_dev_id == -1 && use_reduce == 1) {
            copy_U = (ptiMatrix **)malloc(nt * sizeof(ptiMatrix*));
            for(int t=0; t<nt; ++t) {
                copy_U[t] = (ptiMatrix *)malloc(sizeof(ptiMatrix));
                ptiAssert(ptiNewMatrix(copy_U[t], X.ndims[mode], R) == 0);
                ptiAssert(ptiConstantMatrix(copy_U[t], 0) == 0);
            }
            ptiNnzIndex bytes = nt * X.ndims[mode] * R * sizeof(ptiValue);
            bytestr = ptiBytesString(bytes);
            printf("MODE MATRIX COPY=%s\n", bytestr);
            free(bytestr);
        }

        ptiIndex * mats_order = (ptiIndex*)malloc(nmodes * sizeof(ptiIndex));
        switch (sortcase) {
        case 0:
        case 3:
        case 4:
            mats_order[0] = mode;
            for(ptiIndex i=1; i<nmodes; ++i)
                mats_order[i] = (mode+i) % nmodes;
            break;
        case 1: // Reverse of mode_order except the 1st one
            mats_order[0] = mode;
            for(ptiIndex i=1; i<nmodes; ++i)
                mats_order[i] = mode_order[nmodes - i];
            break;
        case 2: // Totally reverse of mode_order
            for(ptiIndex i=0; i<nmodes; ++i)
                mats_order[i] = mode_order[nmodes - i];
            break;
        }
        // printf("mats_order:\n");
        // ptiDumpIndexArray(mats_order, nmodes, stdout);

        /* For warm-up caches, timing not included */
        if(cuda_dev_id == -2) {
            nthreads = 1;
            ptiAssert(ptiMTTKRP(&X, U, mats_order, mode) == 0);
        } else if(cuda_dev_id == -1) {
            printf("nt: %d\n", nt);
            if(use_reduce == 1) {
                printf("ptiOmpMTTKRP_Reduce:\n");
                ptiAssert(ptiOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
            } else {
                printf("ptiOmpMTTKRP:\n");
                ptiAssert(ptiOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
                // printf("ptiOmpMTTKRP_Lock:\n");
                // ptiAssert(ptiOmpMTTKRP_Lock(&X, U, mats_order, mode, nt, lock_pool) == 0);
            }
        }

        
        ptiTimer timer;
        ptiNewTimer(&timer, 0);
        ptiStartTimer(timer);

        for(int it=0; it<niters; ++it) {
            // ptiAssert(ptiConstantMatrix(U[nmodes], 0) == 0);
            if(cuda_dev_id == -2) {
                nthreads = 1;
                ptiAssert(ptiMTTKRP(&X, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                if(use_reduce == 1) {
                    ptiAssert(ptiOmpMTTKRP_Reduce(&X, U, copy_U, mats_order, mode, nt) == 0);
                } else {
                    ptiAssert(ptiOmpMTTKRP(&X, U, mats_order, mode, nt) == 0);
                    // printf("ptiOmpMTTKRP_Lock:\n");
                    // ptiAssert(ptiOmpMTTKRP_Lock(&X, U, mats_order, mode, nt, lock_pool) == 0);
                }
            }
        }

        ptiStopTimer(timer);

        if(cuda_dev_id == -2 || cuda_dev_id == -1) {
            double aver_time = ptiPrintAverageElapsedTime(timer, niters, "CPU SpTns MTTKRP");

            double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
            uint64_t bytes = ( nmodes * sizeof(ptiIndex) + sizeof(ptiValue) ) * X.nnz;
            for (ptiIndex m=0; m<nmodes; ++m) {
                bytes += X.ndims[m] * R * sizeof(ptiValue);
            }
            double gbw = (double)bytes / aver_time / 1e9;
            printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
        }
        ptiFreeTimer(timer);

        if(fo != NULL) {
            ptiAssert(ptiDumpMatrix(U[nmodes], fo) == 0);
            fclose(fo);
        }

    } // End execute a specified mode

    if(fo != NULL) {
        fclose(fo);
    }
    if(cuda_dev_id == -1) {
        if (use_reduce == 1) {
            for(int t=0; t<nt; ++t) {
                ptiFreeMatrix(copy_U[t]);
            }
            free(copy_U);
        }
        if(lock_pool != NULL) {
            ptiMutexFree(lock_pool);
        }
    }
    for(ptiIndex m=0; m<nmodes; ++m) {
        ptiFreeMatrix(U[m]);
    }
    ptiFreeSparseTensor(&X);
    free(mats_order);
    free(mode_order);
    ptiFreeMatrix(U[nmodes]);
    free(U);

    return 0;
}
