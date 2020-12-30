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
#include "../src/sptensor/hicoo/hicoo.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
    printf("         -e RENUMBER, --renumber=RENUMBER\n");
    printf("         -m MODE, --mode=MODE (default -1: loop all modes)\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t TK, --tk=TK\n");
    printf("         -l TB, --tb=TB\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    printf("mttkrp_hicoo_renumber: \n");

    FILE *fi = NULL, *fo = NULL;
    ptiSparseTensor tsr;
    ptiMatrix ** U;
    ptiSparseTensorHiCOO hitsr;
    ptiElementIndex sb_bits;
    ptiElementIndex sk_bits;
    ptiElementIndex sc_bits;

    ptiIndex mode = HIPARTI_INDEX_MAX;
    ptiIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 0;
    int renumber = 0;
    int niters_renum = 3;
    /* renumber:
     * = 0 : no renumbering.
     * = 1 : renumber with Lexi-order
     * = 2 : renumber with BFS-like
     * = 3 : randomly renumbering, specify niters_renum.
     */
    int tk = 1;
    int tb = 1;
    printf("niters: %d\n", niters);
    int retval;

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", required_argument, 0, 'o'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {"mode", required_argument, 0, 'm'},
            {"impl-num", optional_argument, 0, 'p'},
            {"renumber", optional_argument, 0, 'e'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"tb", optional_argument, 0, 'l'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        // c = getopt_long(argc, argv, "i:o:b:k:c:m:", long_options, &option_index);
        c = getopt_long(argc, argv, "i:o:b:k:c:m:p:e:d:r:t:l:n:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            ptiAssert(fi != NULL);
            break;
        case 'o':
            fo = fopen(optarg, "aw");
            ptiAssert(fo != NULL);
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
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &tk);
            break;
        case 'l':
            sscanf(optarg, "%d", &tb);
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

    ptiAssert(ptiLoadSparseTensor(&tsr, 1, fi) == 0);
    // ptiSparseTensorSortIndex(&tsr, 1);
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
            orderit(&tsr, map_inds, renumber, niters_renum);
            // ptiIndexRenumber(&tsr, map_inds, renumber, niters_renum);
        }
        if ( renumber == 3) { /* Set randomly renumbering */
            printf("[Random Indexing]\n");        
            ptiGetRandomShuffledIndices(&tsr, map_inds);
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


        // ptiSparseTensorSortIndex(&tsr, 1);
        // printf("map_inds:\n");
        // for(ptiIndex m = 0; m < tsr.nmodes; ++m) {
        //     ptiDumpIndexArray(map_inds[m], tsr.ndims[m], stdout);
        // }
        // ptiAssert(ptiDumpSparseTensor(&tsr, 0, stdout) == 0);
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
    U = (ptiMatrix **)malloc((nmodes+1) * sizeof(ptiMatrix*));
    for(ptiIndex m=0; m<nmodes+1; ++m) {
      U[m] = (ptiMatrix *)malloc(sizeof(ptiMatrix));
    }
    ptiIndex max_ndims = 0;
    for(ptiIndex m=0; m<nmodes; ++m) {
      ptiAssert(ptiNewMatrix(U[m], hitsr.ndims[m], R) == 0);
      ptiAssert(ptiConstantMatrix(U[m], 1) == 0);
      // ptiAssert(ptiRandomizeMatrix(U[m]) == 0);
      if(hitsr.ndims[m] > max_ndims)
        max_ndims = hitsr.ndims[m];
      factor_bytes += hitsr.ndims[m] * R * sizeof(ptiValue);
      // ptiAssert(ptiDumpMatrix(U[m], stdout) == 0);
    }
    ptiAssert(ptiNewMatrix(U[nmodes], max_ndims, R) == 0);
    ptiAssert(ptiConstantMatrix(U[nmodes], 0) == 0);
    // ptiAssert(ptiDumpMatrix(U[nmodes], stdout) == 0);

    /* output factor size */
    char * bytestr;
    bytestr = ptiBytesString(factor_bytes);
    printf("FACTORS-STORAGE=%s\n", bytestr);
    printf("\n");
    free(bytestr);

    ptiIndex * mats_order = (ptiIndex*)malloc(nmodes * sizeof(*mats_order));

    if (mode == HIPARTI_INDEX_MAX) {
        for(ptiIndex mode=0; mode<nmodes; ++mode) {
            /* Reset U[nmodes] */
            U[nmodes]->nrows = hitsr.ndims[mode];
            ptiAssert(ptiConstantMatrix(U[nmodes], 0) == 0);

            mats_order[0] = mode;
            for(ptiIndex i=1; i<nmodes; ++i)
                mats_order[i] = (mode+i) % nmodes;
            // printf("mats_order:\n");
            // ptiDumpIndexArray(mats_order, nmodes, stdout);

            /* For warm-up caches, timing not included */
            if(cuda_dev_id == -2) {
                nthreads = 1;
                ptiAssert(ptiMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                printf("tk: %d, tb: %d\n", tk, tb);
                ptiAssert(ptiOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, tk, tb) == 0);
            }

            ptiTimer timer;
            ptiNewTimer(&timer, 0);
            ptiStartTimer(timer);

            for(int it=0; it<niters; ++it) {
                if(cuda_dev_id == -2) {
                    nthreads = 1;
                    ptiAssert(ptiMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
                } else if(cuda_dev_id == -1) {
                    /* Atomic implementation */
                    ptiAssert(ptiOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, tk, tb) == 0);
                }
            }

            ptiStopTimer(timer);
            char * prg_name;
            asprintf(&prg_name, "CPU  SpTns MTTKRP MODE %"HIPARTI_PRI_INDEX, mode);
            ptiPrintAverageElapsedTime(timer, niters, prg_name);
            printf("\n");
            ptiFreeTimer(timer);

            if(fo != NULL) {
                if (renumber > 0) {
                    ptiMatrixInverseShuffleIndices(U[nmodes], map_inds[mode]);
                }
                ptiAssert(ptiDumpMatrix(U[nmodes], fo) == 0);
            }

        }   // End nmodes

    } else {

        mats_order[0] = mode;
        for(ptiIndex i=1; i<nmodes; ++i)
            mats_order[i] = (mode+i) % nmodes;
        // printf("mats_order:\n");
        // ptiDumpIndexArray(mats_order, nmodes, stdout);

        /* For warm-up caches, timing not included */
        if(cuda_dev_id == -2) {
            nthreads = 1;
            ptiAssert(ptiMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
        } else if(cuda_dev_id == -1) {
            printf("tk: %d, tb: %d\n", tk, tb);
            ptiAssert(ptiOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, tk, tb) == 0);
        }

        ptiTimer timer;
        ptiNewTimer(&timer, 0);
        ptiStartTimer(timer);

        for(int it=0; it<niters; ++it) {
            if(cuda_dev_id == -2) {
                nthreads = 1;
                ptiAssert(ptiMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
            } else if(cuda_dev_id == -1) {
                ptiAssert(ptiOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, tk, tb) == 0);
            }
        }

        ptiStopTimer(timer);
        ptiPrintAverageElapsedTime(timer, niters, "CPU  SpTns MTTKRP");
        ptiFreeTimer(timer);

        if(fo != NULL) {
            if (renumber > 0) {
                ptiMatrixInverseShuffleIndices(U[nmodes], map_inds[mode]);
            }
            ptiAssert(ptiDumpMatrix(U[nmodes], fo) == 0);
        }
    }   // End execute a specified mode

    if(fo != NULL) {
        fclose(fo);
    }
    if (renumber > 0) {
        for(ptiIndex m = 0; m < tsr.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }
    for(ptiIndex m=0; m<nmodes; ++m) {
        ptiFreeMatrix(U[m]);
    }
    free(mats_order);
    ptiFreeMatrix(U[nmodes]);
    free(U);
    ptiFreeSparseTensorHiCOO(&hitsr);

    return 0;
}
