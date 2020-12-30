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

int main(int argc, char * const argv[]) {
    char * mm_filename = NULL;
    FILE *fi = NULL, *fo = NULL;
    ptiSparseMatrix spA;
    ptiMatrix B, C;
    ptiIndex R = 16;
    int niters = 5;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    /* OpenMP */
    int cuda_dev_id = -2;
    int nthreads = 1;   // get from OMP_NUM_THREADS environment
    int use_reduce = 1; // privatization or not
    ptiMatrix * Cbufs;

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"rank", optional_argument, 0, 'r'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {"use-reduce", optional_argument, 0, 'u'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:r:d:u:", long_options, &option_index);
        if(c == -1) {
            break;
        }

        switch(c) {
        case 'i':
            mm_filename = optarg;
            fi = fopen(optarg, "r");
            ptiAssert(fi != NULL);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            ptiAssert(fo != NULL);
            break;
        case 'r':
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &R);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'u':
            sscanf(optarg, "%d", &use_reduce);
            break;
        default:
            abort();
        }
    }
    printf("B ncols: %d\n", R);
    printf("niters: %d\n", niters);
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    if(cuda_dev_id == -1) {
        printf("use_reduce: %d\n", use_reduce);
#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel
        nthreads = omp_get_num_threads();
#endif
        printf("nthreads: %d\n", nthreads);
    }


    if(optind > argc || argc < 2) {
        printf("Usage: %s\n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("         -R RANK\n");
        printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
        printf("         -u use_reduce, --ur=use_reduce\n");
        printf("\n");
        return 1;
    }

    printf("Reading sparse matrix from file (%s) ...",mm_filename);
    fflush(stdout);
    ptiAssert(ptiLoadSparseMatrix(&spA, 1, fi) == 0);
    fclose(fi);
    printf(" done\n");
    ptiSparseMatrixStatus(&spA, stdout);
    // ptiAssert(ptiDumpSparseMatrix(&spA, 0, stdout) == 0);

    ptiNewMatrix(&B, spA.ncols, R);
    ptiRandomizeMatrix(&B);
    ptiNewMatrix(&C, spA.nrows, R);
    ptiConstantMatrix(&C, 0);
    // ptiAssert(ptiDumpMatrix(&B, stdout) == 0);
    // ptiAssert(ptiDumpMatrix(&C, stdout) == 0);

    /* Allocate extra buffer for privatization implementation */
    if(cuda_dev_id == -1 && use_reduce == 1) {
        Cbufs = (ptiMatrix *) malloc(nthreads * sizeof(ptiMatrix));
        for(int t=0; t<nthreads; ++t) {
            ptiNewMatrix(&Cbufs[t], spA.nrows, R);
            ptiConstantMatrix(&Cbufs[t], 0);
        }
        ptiNnzIndex bytes = nthreads * spA.nrows * R * sizeof(ptiValue);
        char * bytestr = ptiBytesString(bytes);
        printf("MATRIX BUFFER=%s\n", bytestr);
        free(bytestr);
    }

    // Warm-up
    if(cuda_dev_id == -2) {
        printf("Run ptiSparseMatrixMulMatrix:\n");
        ptiSparseMatrixMulMatrix(&C, &spA, &B);
    } else if(cuda_dev_id == -1) {
        if(use_reduce == 1) {
            printf("Run ptiOmpSparseMatrixMulMatrix_Reduce:\n");
            ptiOmpSparseMatrixMulMatrix_Reduce(&C, Cbufs, &spA, &B);
        } else {
            printf("Run ptiOmpSparseMatrixMulMatrix:\n");
            ptiOmpSparseMatrixMulMatrix(&C, &spA, &B);
        }
    }

    ptiStartTimer(timer);
    for(int i=0; i<niters; ++i) {
        if(cuda_dev_id == -2) {
            ptiSparseMatrixMulMatrix(&C, &spA, &B);
        } else if(cuda_dev_id == -1) {
            if(use_reduce == 1) {
                ptiOmpSparseMatrixMulMatrix_Reduce(&C, Cbufs, &spA, &B);
            } else {
                ptiOmpSparseMatrixMulMatrix(&C, &spA, &B);
            }
        }
    }
    ptiStopTimer(timer);
    printf("\n");
    double elapsed_time = ptiPrintAverageElapsedTime(timer, niters, "COO-SpMM");
    ptiNnzIndex flops = 2 * spA.nnz * R;
    ptiPrintGFLOPS(elapsed_time, flops, "COO-SpMM");

    if(fo != NULL) {
        ptiAssert(ptiDumpMatrix(&C, fo) == 0);
        fclose(fo);
    }

    if(cuda_dev_id == -1 && use_reduce == 1) {
        for(int t=0; t<nthreads; ++t) {
            ptiFreeMatrix(&Cbufs[t]);
        }
    }
    ptiFreeSparseMatrix(&spA);
    ptiFreeMatrix(&B);
    ptiFreeMatrix(&C);
    ptiFreeTimer(timer);

    return 0;
}
