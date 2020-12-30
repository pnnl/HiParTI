/*
    This file is part of HiParTI!.

    HiParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    HiParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with HiParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <HiParTI.h>

int main(int argc, char * const argv[]) {
    FILE *fi = NULL, *fo = NULL;
    ptiSparseMatrix mtx;
    ptiSparseMatrixCSR csrmtx;
    ptiValueVector x, y;
    int niters = 50;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    /* OpenMP */
    int cuda_dev_id = -2;
    int nthreads = 1;   // get from OMP_NUM_THREADS environment

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:d:", long_options, &option_index);
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
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        default:
            abort();
        }
    }
    printf("niters: %d\n", niters);
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    if(cuda_dev_id == -1) {
#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel
        nthreads = omp_get_num_threads();
#endif
        printf("nthreads: %d\n", nthreads);
    }

    if(optind > argc || argc < 3) {
        printf("Usage: %s\n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
        printf("\n");
        return 1;
    }

    /// Load sparse matrix in COO format
    ptiAssert(ptiLoadSparseMatrix(&mtx, 1, fi) == 0);
    fclose(fi);
    ptiRandomValueVector(&(mtx.values));    // to better compare results
    ptiSparseMatrixStatus(&mtx, stdout);
    // ptiAssert(ptiDumpSparseMatrix(&mtx, 0, stdout) == 0);

    /// Convert sparse matrix to CSR format
    ptiAssert(ptiSparseMatrixToCSR(&csrmtx, &mtx) == 0);
    ptiFreeSparseMatrix(&mtx);
    ptiSparseMatrixStatusCSR(&csrmtx, stdout);
    // ptiAssert(ptiDumpSparseMatrixCSR(&csrmtx, stdout) == 0);

    /// Initialize values for vectors x and y
    ptiNewValueVector(&x, csrmtx.ncols, csrmtx.ncols);
    ptiRandomValueVector(&x);
    ptiNewValueVector(&y, csrmtx.nrows, csrmtx.nrows);
    // ptiAssert(ptiDumpValueVector(&x, stdout) == 0);
    // ptiAssert(ptiDumpValueVector(&y, stdout) == 0);

#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel
    nthreads = omp_get_num_threads();
    printf("nthreads: %d\n", nthreads);
#endif
    
    // Warm-up
    if(cuda_dev_id == -2) {
        printf("Run ptiSparseMatrixMulVectorCSR:\n");
        ptiSparseMatrixMulVectorCSR(&y, &csrmtx, &x);
    } else if(cuda_dev_id == -1) {
        printf("Run ptiOmpSparseMatrixMulVectorCSR:\n");
        ptiOmpSparseMatrixMulVectorCSR(&y, &csrmtx, &x);
    }
    

    ptiStartTimer(timer);
    for(int i=0; i<niters; ++i) {
        if(cuda_dev_id == -2) {
            ptiSparseMatrixMulVectorCSR(&y, &csrmtx, &x);
        } else if(cuda_dev_id == -1) {
            ptiOmpSparseMatrixMulVectorCSR(&y, &csrmtx, &x);
        }
    }
    ptiStopTimer(timer);
    printf("\n");
    double elapsed_time = ptiPrintAverageElapsedTime(timer, niters, "CSR-SpMV");
    ptiNnzIndex flops = 2 * csrmtx.nnz;
    ptiPrintGFLOPS(elapsed_time, flops, "CSR-SpMV");


    if(fo != NULL) {
        ptiAssert(ptiDumpValueVector(&y, fo) == 0);
        fclose(fo);
    }

    ptiFreeSparseMatrixCSR(&csrmtx);
    ptiFreeValueVector(&x);
    ptiFreeValueVector(&y);
    ptiFreeTimer(timer);

    return 0;
}
