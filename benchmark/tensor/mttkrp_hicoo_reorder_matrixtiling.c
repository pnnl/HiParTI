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
#include <HiParTI.h>
#include "../src/sptensor/sptensor.h"
#include "../src/sptensor/hicoo/hicoo.h"
#ifdef HIPARTI_USE_OPENMP
    #include <omp.h>
#endif


static void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -s SHUFFLE FILE, --shuffle=SHUFFLE FILE\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
    printf("         -e RENUMBER, --renumber=RENUMBER\n");
    printf("         -n NITERS_RENUM\n");
    printf("         -m MODE, --mode=MODE (default -1: loop all modes)\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t TK, --tk=TK\n");
    printf("         -l TB, --tb=TB\n");
    printf("         -a balanced\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    printf("mttkrp_hicoo_renumber_matrixtiling: \n");

    FILE *fi = NULL, *fo = NULL, *fs = NULL;
    ptiSparseTensor tsr;
    ptiRankMatrix ** U;
    ptiRankMatrix ** copy_U;
    ptiSparseTensorHiCOO hitsr;
    ptiElementIndex sb_bits;
    ptiElementIndex sk_bits;
    ptiElementIndex sc_bits;

    ptiIndex mode = HIPARTI_INDEX_MAX;
    ptiElementIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 1;
    int renumber = 0;
    int niters_renum = 3;
    /* renumber:
     * = 0 : no renumbering.
     * = 1 : renumber with Lexi-order, specify niters_renum.
     * = 2 : renumber with BFS-like
     * = 3 : randomly renumbering.
     * = 4 : renumbering according to shuffle map in fs.
     */
    int tk = 1;
    int tb = 1;
    int par_iters = 0;
    int balanced = -1, input_balanced = -1;
    printf("niters: %d\n", niters);

    if(argc <= 6) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {"mode", required_argument, 0, 'm'},
            {"shuffle", optional_argument, 0, 's'},
            {"output", optional_argument, 0, 'o'},
            {"impl-num", optional_argument, 0, 'p'},
            {"renumber", optional_argument, 0, 'e'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"tb", optional_argument, 0, 'l'},
            {"balanced", optional_argument, 0, 'a'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        c = getopt_long(argc, argv, "i:o:b:k:c:m:p:e:d:r:t:l:n:a:s:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            ptiAssert(fi != NULL);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            ptiAssert(fo != NULL);
            break;
        case 's':
            fs = fopen(optarg, "r");
            ptiAssert(fs != NULL);
            break;
        case 'b':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 'c':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sc_bits);
            break;
        case 'm':
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &mode);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'e':
            sscanf(optarg, "%d", &renumber);
            break;
        case 'n':
            sscanf(optarg, "%d", &niters_renum);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &tk);
            break;
        case 'l':
            sscanf(optarg, "%d", &tb);
            break;
        case 'a':
            sscanf(optarg, "%d", &input_balanced);
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
    printf("renumber: %d\n", renumber);
    if (renumber == 1)
        printf("niters_renum: %d\n\n", niters_renum);

    /* A sorting included in load tensor */
    ptiAssert(ptiLoadSparseTensor(&tsr, 1, fi) == 0);
    fclose(fi);
    ptiSparseTensorStatus(&tsr, stdout);
    // ptiAssert(ptiDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Renumber the input tensor */
    ptiIndex ** map_inds;
    if (renumber > 0) {
        map_inds = (ptiIndex **)malloc(tsr.nmodes * sizeof *map_inds);
        pti_CheckOSError(!map_inds, "MTTKRP HiCOO");
        for(ptiIndex m = 0; m < tsr.nmodes; ++m) {
            map_inds[m] = (ptiIndex *)malloc(tsr.ndims[m] * sizeof (ptiIndex));
            pti_CheckError(!map_inds[m], "MTTKRP HiCOO", NULL);
            for(ptiIndex i = 0; i < tsr.ndims[m]; ++i)
                map_inds[m][i] = i;
        }

        ptiTimer renumber_timer;
        ptiNewTimer(&renumber_timer, 0);
        ptiStartTimer(renumber_timer);

        if ( renumber == 1 || renumber == 2) { /* Set the Lexi-order or BFS-like renumbering */
            #if 0
            orderit(&tsr, map_inds, renumber, niters_renum);
            #else
            ptiIndexRenumber(&tsr, map_inds, renumber, niters_renum, sb_bits, tk, impl_num);
            #endif

        } else if ( renumber == 3) { /* Set randomly renumbering */
            ptiGetRandomShuffledIndices(&tsr, map_inds);
        } else if ( renumber == 4) { /* read shuffle map from fs */
            if (fs == NULL) {
                printf("[Error]: Input shuffle file.\n");
                return -1;
            }
            ptiLoadShuffleFile(&tsr, fs, map_inds);
            ptiSparseTensorInvMap(&tsr, map_inds);
            fclose(fs);
        }
        fflush(stdout);

        ptiStopTimer(renumber_timer);
        ptiPrintElapsedTime(renumber_timer, "Renumbering");
        ptiFreeTimer(renumber_timer);

        ptiTimer shuffle_timer;
        ptiNewTimer(&shuffle_timer, 0);
        ptiStartTimer(shuffle_timer);

        ptiSparseTensorShuffleIndices(&tsr, map_inds);

        ptiStopTimer(shuffle_timer);
        ptiPrintElapsedTime(shuffle_timer, "Shuffling time");
        ptiFreeTimer(shuffle_timer);
        printf("\n");


        // ptiSparseTensorSortIndex(&tsr, 1);   // debug purpose only
        // FILE * debug_fp = fopen("new.txt", "w");
        // ptiAssert(ptiDumpSparseTensor(&tsr, 0, debug_fp) == 0);
        // fprintf(stdout, "\nmap_inds:\n");
        // for(ptiIndex m = 0; m < tsr.nmodes; ++m) {
        //     ptiDumpIndexArray(map_inds[m], tsr.ndims[m], stdout);
        // }
        // fclose(debug_fp);
    }

    /* Convert to HiCOO tensor */
    ptiNnzIndex max_nnzb = 0;
    ptiTimer convert_timer;
    ptiNewTimer(&convert_timer, 0);
    ptiStartTimer(convert_timer);

    ptiAssert(ptiSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, sc_bits, tk) == 0);

    ptiStopTimer(convert_timer);
    ptiPrintElapsedTime(convert_timer, "Convert HiCOO");
    ptiFreeTimer(convert_timer);

    ptiFreeSparseTensor(&tsr);
    ptiSparseTensorStatusHiCOO(&hitsr, stdout);
    // ptiAssert(ptiDumpSparseTensorHiCOO(&hitsr, stdout) == 0);

    /* Initialize factor matrices */
    ptiIndex nmodes = hitsr.nmodes;
    ptiNnzIndex factor_bytes = 0;
    U = (ptiRankMatrix **)malloc((nmodes+1) * sizeof(ptiRankMatrix*));
    for(ptiIndex m=0; m<nmodes+1; ++m) {
      U[m] = (ptiRankMatrix *)malloc(sizeof(ptiRankMatrix));
    }
    ptiIndex max_ndims = 0;
    for(ptiIndex m=0; m<nmodes; ++m) {
      ptiAssert(ptiNewRankMatrix(U[m], hitsr.ndims[m], R) == 0);
      ptiAssert(ptiConstantRankMatrix(U[m], 1) == 0);
      // ptiAssert(ptiRandomizeMatrix(U[m]) == 0);
      if(hitsr.ndims[m] > max_ndims)
        max_ndims = hitsr.ndims[m];
      factor_bytes += hitsr.ndims[m] * R * sizeof(ptiValue);
      // ptiAssert(ptiDumpRankMatrix(U[m], stdout) == 0);
    }
    ptiAssert(ptiNewRankMatrix(U[nmodes], max_ndims, R) == 0);
    ptiAssert(ptiConstantRankMatrix(U[nmodes], 0) == 0);
    // ptiAssert(ptiDumpRankMatrix(U[nmodes], stdout) == 0);

    /* output factor size */
    char * bytestr;
    bytestr = ptiBytesString(factor_bytes);
    printf("FACTORS-STORAGE=%s\n", bytestr);
    printf("\n");
    free(bytestr);

    ptiIndex * mats_order = (ptiIndex*)malloc(nmodes * sizeof(*mats_order));
    ptiIndex sk = (ptiIndex)pow(2, hitsr.sk_bits);

    if (mode == HIPARTI_INDEX_MAX) {
        for(ptiIndex mode=0; mode < nmodes; ++mode) {
            par_iters = 0;
            /* Reset U[nmodes] */
            U[nmodes]->nrows = hitsr.ndims[mode];
            ptiAssert(ptiConstantRankMatrix(U[nmodes], 0) == 0);

            /* Determine use balanced or not */
            double rest_nnz_portion_th = 0.10;
            printf("rest_nnz_portion_th: %.2lf\n", rest_nnz_portion_th);
            ptiIndex num_kernel_dim = (hitsr.ndims[mode] + sk - 1) / sk;
            ptiIndex npars = hitsr.kschr_balanced_pos[mode][0].len - 1;
            ptiNnzIndex sum_balanced_nnzk = 0;
            ptiIndex kernel_ndim = (hitsr.ndims[mode] + sk - 1)/sk;
            for(ptiIndex i=0; i < kernel_ndim; ++i) {
              for(ptiIndex j=0; j < hitsr.kschr_balanced[mode][i].len; ++j) {
                ptiIndex kernel_num = hitsr.kschr_balanced[mode][i].data[j];
                sum_balanced_nnzk += hitsr.knnzs.data[kernel_num];
              }
            }
            double rest_nnz_portion = 1.0 - (double)sum_balanced_nnzk / hitsr.nnz;
            if (input_balanced == -1) {
                if (rest_nnz_portion < rest_nnz_portion_th && max(num_kernel_dim, npars) > PAR_MIN_DEGREE * NUM_CORES ) {
                    balanced = 1;
                } else {
                    balanced = 0;
                }
            } else {
                balanced = input_balanced;
            }
            printf("balanced: %d\n", balanced);

            /* determine niters or num_kernel_dim to be parallelized */
            ptiIndex ratio_schr_ncols, num_tasks;
            if(balanced == 0) {
                ratio_schr_ncols = hitsr.nkiters[mode] / num_kernel_dim;
            } else if (balanced == 1) {
                ratio_schr_ncols = npars / num_kernel_dim;
            }
            printf("Schr ncols / nrows: %u (threshold: %u)\n", ratio_schr_ncols, PAR_DEGREE_REDUCE);
            if(num_kernel_dim <= PAR_MIN_DEGREE * NUM_CORES && ratio_schr_ncols >= PAR_DEGREE_REDUCE) {
                par_iters = 1;
            }
            if(balanced == 0) {
                num_tasks = (par_iters == 1) ? hitsr.nkiters[mode] : num_kernel_dim;
            } else if (balanced == 1) {
                num_tasks = (par_iters == 1) ? npars : num_kernel_dim;
            }
            printf("par_iters: %d, num_tasks: %u\n", par_iters, num_tasks);

            /* Set zeros for temporary copy_U, for mode-"mode" */
            if(cuda_dev_id == -1 && par_iters == 1) {
                copy_U = (ptiRankMatrix **)malloc(tk * sizeof(ptiRankMatrix*));
                for(int t=0; t<tk; ++t) {
                    copy_U[t] = (ptiRankMatrix *)malloc(sizeof(ptiRankMatrix));
                    ptiAssert(ptiNewRankMatrix(copy_U[t], hitsr.ndims[mode], R) == 0);
                    ptiAssert(ptiConstantRankMatrix(copy_U[t], 0) == 0);
                }
                ptiNnzIndex bytes = tk * hitsr.ndims[mode] * R * sizeof(ptiValue);
                bytestr = ptiBytesString(bytes);
                printf("MODE MATRIX COPY=%s\n", bytestr);
            }

            mats_order[0] = mode;
            for(ptiIndex i=1; i<nmodes; ++i)
                mats_order[i] = (mode+i) % nmodes;
            // printf("mats_order:\n");
            // ptiDumpIndexArray(mats_order, nmodes, stdout);

            /* For warm-up caches, timing not included */
            if(cuda_dev_id == -2) {
                nthreads = 1;
                ptiAssert(ptiMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                printf("tk: %d, tb: %d\n", tk, tb);
                // printf("ptiOmpMTTKRPHiCOO_MatrixTiling:\n");
                // ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk, tb) == 0);
                if(par_iters == 0) {
                    printf("ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled:\n");
                    ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, tb, balanced) == 0);
                } else {
                    printf("ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce:\n");
                    ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, tb, balanced) == 0);
                }
            }

            ptiTimer timer;
            ptiNewTimer(&timer, 0);
            ptiStartTimer(timer);

            for(int it=0; it<niters; ++it) {
                if(cuda_dev_id == -2) {
                    nthreads = 1;
                    ptiAssert(ptiMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
                } else if(cuda_dev_id == -1) {
                    // ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk, tb) == 0);
                    if(par_iters == 0) {
                        ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, tb, balanced) == 0);
                    } else {
                        ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, tb, balanced) == 0);
                    }
                }
            }

            ptiStopTimer(timer);
            char * prg_name;
            asprintf(&prg_name, "CPU  SpTns MTTKRP MODE %"HIPARTI_PRI_INDEX, mode);
            ptiPrintAverageElapsedTime(timer, niters, prg_name);
            printf("\n");
            ptiFreeTimer(timer);


            if (renumber > 0) {
                ptiTimer back_shuffle_timer;
                ptiNewTimer(&back_shuffle_timer, 0);
                ptiStartTimer(back_shuffle_timer);
            
                ptiRankMatrixInverseShuffleIndices(U[nmodes], map_inds[mode]);

                ptiStopTimer(back_shuffle_timer);
                ptiPrintElapsedTime(back_shuffle_timer, "Inverse-Shuffling time");
                ptiFreeTimer(back_shuffle_timer);
                printf("\n");
            }
            if(fo != NULL) {
                ptiAssert(ptiDumpRankMatrix(U[nmodes], fo) == 0);
            }

        }   // End nmodes

    } else {
    
        /* determine niters or num_kernel_dim to be parallelized */
        ptiIndex sk = (ptiIndex)pow(2, hitsr.sk_bits);
        ptiIndex num_kernel_dim = (hitsr.ndims[mode] + sk - 1) / sk;
        printf("num_kernel_dim: %u, hitsr.nkiters[mode] / num_kernel_dim: %u\n", num_kernel_dim, hitsr.nkiters[mode]/num_kernel_dim);
        if(num_kernel_dim <= PAR_MIN_DEGREE * NUM_CORES && hitsr.nkiters[mode] / num_kernel_dim >= PAR_DEGREE_REDUCE) {
            par_iters = 1;
        }

        /* Set zeros for temporary copy_U, for mode-"mode" */
        if(cuda_dev_id == -1 && par_iters == 1) {
            copy_U = (ptiRankMatrix **)malloc(tk * sizeof(ptiRankMatrix*));
            for(int t=0; t<tk; ++t) {
                copy_U[t] = (ptiRankMatrix *)malloc(sizeof(ptiRankMatrix));
                ptiAssert(ptiNewRankMatrix(copy_U[t], hitsr.ndims[mode], R) == 0);
                ptiAssert(ptiConstantRankMatrix(copy_U[t], 0) == 0);
            }
            ptiNnzIndex bytes = tk * hitsr.ndims[mode] * R * sizeof(ptiValue);
            bytestr = ptiBytesString(bytes);
            printf("MODE MATRIX COPY=%s\n", bytestr);
        }

        ptiIndex * mats_order = (ptiIndex*)malloc(nmodes * sizeof(*mats_order));
        mats_order[0] = mode;
        for(ptiIndex i=1; i<nmodes; ++i)
            mats_order[i] = (mode+i) % nmodes;
        // printf("mats_order:\n");
        // ptiDumpIndexArray(mats_order, nmodes, stdout);

        /* For warm-up caches, timing not included */
        if(cuda_dev_id == -2) {
            nthreads = 1;
            ptiAssert(ptiMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
        } else if(cuda_dev_id == -1) {
            printf("tk: %d, tb: %d\n", tk, tb);
            // printf("ptiOmpMTTKRPHiCOO_MatrixTiling:\n");
            // ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk, tb) == 0);
            if(par_iters == 0) {
                printf("ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled:\n");
                ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, tb, balanced) == 0);
            } else {
                printf("ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce:\n");
                ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, tb, balanced) == 0);
            }
        }

        ptiTimer timer;
        ptiNewTimer(&timer, 0);
        ptiStartTimer(timer);

        for(int it=0; it<niters; ++it) {
            if(cuda_dev_id == -2) {
                nthreads = 1;
                ptiAssert(ptiMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                // ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling(&hitsr, U, mats_order, mode, tk, tb) == 0);
                if(par_iters == 0) {
                    ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&hitsr, U, mats_order, mode, tk, tb, balanced) == 0);
                } else {
                    ptiAssert(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&hitsr, U, copy_U, mats_order, mode, tk, tb, balanced) == 0);
                }
            }
        }

        ptiStopTimer(timer);
        ptiPrintAverageElapsedTime(timer, niters, "CPU  SpTns MTTKRP");
        printf("\n");
        ptiFreeTimer(timer);

        ptiTimer shuffle_timer;
        ptiNewTimer(&shuffle_timer, 0);
        ptiStartTimer(shuffle_timer);
        if (renumber > 0) {
            ptiRankMatrixInverseShuffleIndices(U[nmodes], map_inds[mode]);
        }
        ptiStopTimer(shuffle_timer);
        ptiPrintElapsedTime(shuffle_timer, "Inverse-Shuffling time");
        ptiFreeTimer(shuffle_timer);

        if(fo != NULL) {
            ptiAssert(ptiDumpRankMatrix(U[nmodes], fo) == 0);
        }
    }   // End execute a specified mode

    if (renumber > 0) {
        for(ptiIndex m = 0; m < tsr.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }

    if(fo != NULL) {
        fclose(fo);
    }
    if(cuda_dev_id == -1 && par_iters == 1) {
        for(int t=0; t<tk; ++t) {
            ptiFreeRankMatrix(copy_U[t]);
        }
        free(copy_U);
        free(bytestr);
    }
    for(ptiIndex m=0; m<nmodes; ++m) {
        ptiFreeRankMatrix(U[m]);
    }
    ptiFreeRankMatrix(U[nmodes]);
    free(U);
    free(mats_order);
    ptiFreeSparseTensorHiCOO(&hitsr);

    return 0;
}
