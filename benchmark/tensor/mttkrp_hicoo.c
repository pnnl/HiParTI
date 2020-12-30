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

void print_usage(int argc, char ** argv) {
    printf("Usage: %s [options] \n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
    printf("         -m MODE, --mode=MODE\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t TK, --tk=TK\n");
    printf("         -l TB, --tb=TB\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    FILE *fi = NULL, *fo = NULL;
    ptiSparseTensor tsr;
    ptiMatrix ** U;
    ptiSparseTensorHiCOO hitsr;
    ptiElementIndex sb_bits;
    ptiElementIndex sk_bits;
    ptiElementIndex sc_bits;

    ptiIndex mode = 0;
    ptiIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads;
    int impl_num = 0;
    int tk = 1;
    int tb = 1;
    printf("niters: %d\n", niters);
    int retval;

    if(argc <= 3) { // #Required arguments
        print_usage(argc, argv);
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
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"tb", optional_argument, 0, 'l'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        // c = getopt_long(argc, argv, "i:o:b:k:c:m:", long_options, &option_index);
        c = getopt_long(argc, argv, "i:o:b:k:c:m:p:d:r:t:l:", long_options, &option_index);
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
            print_usage(argc, argv);
            exit(1);
        }
    }


    ptiAssert(ptiLoadSparseTensor(&tsr, 1, fi) == 0);
    // ptiSparseTensorSortIndex(&tsr, 1);
    fclose(fi);
    ptiSparseTensorStatus(&tsr, stdout);
    // ptiAssert(ptiDumpSparseTensor(&tsr, 0, stdout) == 0);

    ptiTimer convert_timer;
    ptiNewTimer(&convert_timer, 0);
    ptiStartTimer(convert_timer);

    /* Convert to HiCOO tensor */
    ptiNnzIndex max_nnzb = 0;
    ptiAssert(ptiSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, sc_bits, tk) == 0);
    ptiFreeSparseTensor(&tsr);
    ptiSparseTensorStatusHiCOO(&hitsr, stdout);
    // ptiAssert(ptiDumpSparseTensorHiCOO(&hitsr, stdout) == 0);

    ptiStopTimer(convert_timer);
    ptiPrintElapsedTime(convert_timer, "Convert HiCOO");
    ptiFreeTimer(convert_timer);

    ptiIndex nmodes = hitsr.nmodes;
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
      // ptiAssert(ptiDumpMatrix(U[m], stdout) == 0);
    }
    ptiAssert(ptiNewMatrix(U[nmodes], max_ndims, R) == 0);
    ptiAssert(ptiConstantMatrix(U[nmodes], 0) == 0);
    // ptiAssert(ptiDumpMatrix(U[nmodes], stdout) == 0);


    ptiIndex * mats_order = (ptiIndex*)malloc(nmodes * sizeof(*mats_order));
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
        ptiAssert(ptiDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
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
