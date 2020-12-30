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
    ptiSparseMatrixCSR csrA;
    ptiMatrix B, C;
    ptiIndex R = 16;
    int niters = 5;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    /* OpenMP */
    int cuda_dev_id = -2;
    int nthreads = 1;   // get from OMP_NUM_THREADS environment

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"R", optional_argument, 0, 'r'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:r:d:", long_options, &option_index);
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
        default:
            abort();
        }
    }
    printf("B ncols: %d\n", R);
    printf("niters: %d\n", niters);
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    if(cuda_dev_id == -1) {
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

    ptiAssert(ptiSparseMatrixToCSR(&csrA, &spA) == 0);
    ptiFreeSparseMatrix(&spA);
    ptiSparseMatrixStatusCSR(&csrA, stdout);
    // ptiAssert(ptiDumpSparseMatrixCSR(&csrA, stdout) == 0);

    ptiNewMatrix(&B, csrA.ncols, R);
    ptiRandomizeMatrix(&B);
    ptiNewMatrix(&C, csrA.nrows, R);
    ptiConstantMatrix(&C, 0);
    // ptiAssert(ptiDumpMatrix(&B, stdout) == 0);
    // ptiAssert(ptiDumpMatrix(&C, stdout) == 0);


    // Warm-up
    if(cuda_dev_id == -2) {
        printf("Run ptiSparseMatrixMulMatrixCSR:\n");
        ptiSparseMatrixMulMatrixCSR(&C, &csrA, &B);
    } else if(cuda_dev_id == -1) {
        printf("Run ptiOmpSparseMatrixMulMatrixCSR:\n");
        ptiOmpSparseMatrixMulMatrixCSR(&C, &csrA, &B);
    }

    ptiStartTimer(timer);
    for(int i=0; i<niters; ++i) {
        if(cuda_dev_id == -2) {
            ptiSparseMatrixMulMatrixCSR(&C, &csrA, &B);
        } else if(cuda_dev_id == -1) {
            ptiOmpSparseMatrixMulMatrixCSR(&C, &csrA, &B);
        }
    }
    ptiStopTimer(timer);
    printf("\n");
    double elapsed_time = ptiPrintAverageElapsedTime(timer, niters, "CSR-SpMM");
    ptiNnzIndex flops = 2 * csrA.nnz * R;
    ptiPrintGFLOPS(elapsed_time, flops, "CSR-SpMM");

    if(fo != NULL) {
        ptiAssert(ptiDumpMatrix(&C, fo) == 0);
        fclose(fo);
    }

    ptiFreeSparseMatrixCSR(&csrA);
    ptiFreeMatrix(&B);
    ptiFreeMatrix(&C);
    ptiFreeTimer(timer);

    return 0;
}
