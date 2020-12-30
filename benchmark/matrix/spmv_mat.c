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


int main(int argc, char * const argv[]) 
{
    char * mm_filename = NULL;
    FILE *fi = NULL, *fo = NULL;
    ptiSparseMatrix mtx;
    ptiValueVector x, y;
    int niters = 50;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    /* OpenMP */
    int cuda_dev_id = -2;
    int nthreads = 1;   // get from OMP_NUM_THREADS environment
    int use_reduce = 1; // privatization or not
    ptiValueVector * ybufs;

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {"use-reduce", optional_argument, 0, 'u'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:d:u:", long_options, &option_index);
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

    if(optind > argc || argc < 3) {
        printf("Usage: %s\n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
        printf("         -u use_reduce, --ur=use_reduce\n");
        printf("\n");
        return 1;
    }

    /// Load sparse matrix in COO format
    printf("Reading sparse matrix from file (%s) ...",mm_filename);
    fflush(stdout);
    ptiAssert(ptiLoadSparseMatrix(&mtx, 1, fi) == 0);
    fclose(fi);
    printf(" done\n");
    ptiRandomValueVector(&(mtx.values));    // to better compare results
    ptiSparseMatrixStatus(&mtx, stdout);
    // ptiAssert(ptiDumpSparseMatrix(&mtx, 0, stdout) == 0);

    /// Initialize values for vectors x and y
    ptiNewValueVector(&x, mtx.ncols, mtx.ncols);
    ptiRandomValueVector(&x);
    ptiNewValueVector(&y, mtx.nrows, mtx.nrows);
    // ptiAssert(ptiDumpValueVector(&x, stdout) == 0);
    // ptiAssert(ptiDumpValueVector(&y, stdout) == 0);

    /* Allocate extra buffer for privatization implementation */
    if(cuda_dev_id == -1 && use_reduce == 1) {
        ybufs = (ptiValueVector *) malloc(nthreads * sizeof(ptiValueVector));
        for(int t=0; t<nthreads; ++t) {
            ptiNewValueVector(&ybufs[t], mtx.nrows, mtx.nrows);
            ptiConstantValueVector(&ybufs[t], 0);
        }
        ptiNnzIndex bytes = nthreads * mtx.nrows * sizeof(ptiValue);
        char * bytestr = ptiBytesString(bytes);
        printf("VECTOR BUFFER=%s\n", bytestr);
        free(bytestr);
    }


    // Warm-up
    if(cuda_dev_id == -2) {
        printf("Run ptiSparseMatrixMulVector:\n");
        ptiSparseMatrixMulVector(&y, &mtx, &x);
    } else if(cuda_dev_id == -1) {
        if(use_reduce == 1) {
            printf("Run ptiOmpSparseMatrixMulVector_Reduce:\n");
            ptiOmpSparseMatrixMulVector_Reduce(&y, ybufs, &mtx, &x);
        } else {
            printf("Run ptiOmpSparseMatrixMulVector:\n");
            ptiOmpSparseMatrixMulVector(&y, &mtx, &x);
        }
    }
    

    ptiStartTimer(timer);
    for(int i=0; i<niters; ++i) {
        if(cuda_dev_id == -2) {
            ptiSparseMatrixMulVector(&y, &mtx, &x);
        } else if(cuda_dev_id == -1) {
            if(use_reduce == 1) {
                ptiOmpSparseMatrixMulVector_Reduce(&y, ybufs, &mtx, &x);
            } else {
                ptiOmpSparseMatrixMulVector(&y, &mtx, &x);
            }
        }
    }
    ptiStopTimer(timer);
    printf("\n");
    double elapsed_time = ptiPrintAverageElapsedTime(timer, niters, "COO-SpMV");
    ptiNnzIndex flops = 2 * mtx.nnz;
    ptiPrintGFLOPS(elapsed_time, flops, "COO-SpMV");

    if(fo != NULL) {
        ptiAssert(ptiDumpValueVector(&y, fo) == 0);
        fclose(fo);
    }

#ifdef HIPARTI_USE_OPENMP
    if(cuda_dev_id == -1 && use_reduce == 1) {
        for(int t=0; t<nthreads; ++t) {
            ptiFreeValueVector(&ybufs[t]);
        }
        free(ybufs);
    }
#endif
    ptiFreeSparseMatrix(&mtx);
    ptiFreeValueVector(&x);
    ptiFreeValueVector(&y);
    ptiFreeTimer(timer);

    return 0;
}
