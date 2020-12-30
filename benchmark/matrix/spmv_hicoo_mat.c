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
    FILE *fi = NULL, *fo = NULL;
    ptiSparseMatrix mtx;
    ptiSparseMatrixHiCOO himtx;
    ptiValueVector x, y;
    ptiElementIndex sb_bits = 7;
    int niters = 50;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    /* OpenMP */
    int cuda_dev_id = -2;
    int nthreads = 1;   // get from OMP_NUM_THREADS environment
    int use_schedule = 0; // privatization or not
    ptiElementIndex sk_bits = sb_bits;
    int par_iters = 0;  // determine in the code
    ptiValueVector * ybufs;

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"bs", optional_argument, 0, 'b'},
        {"ks", optional_argument, 0, 'k'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {"use-schedule", optional_argument, 0, 'u'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:b:k:d:u:", long_options, &option_index);
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
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'u':
            sscanf(optarg, "%d", &use_schedule);
            break;
        default:
            abort();
        }
    }
    printf("niters: %d\n", niters);
    printf("sb: %ld\n", (long int)pow(2,sb_bits));
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    if(cuda_dev_id == -1) {
        printf("use_schedule: %d\n", use_schedule);
#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel
        nthreads = omp_get_num_threads();
#endif
        printf("sk: %ld\n", (long int)pow(2,sk_bits));
        printf("nthreads: %d\n", nthreads);
    }

    if(optind > argc || argc < 3) {
        printf("Usage: %s\n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
        printf("         -k SUPERBLOCKSIZE (bits), --kernelsize=SUPERBLOCKSIZE (bits)\n");
        printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
        printf("         -u use_schedule, --ur=use_schedule\n");
        printf("\n");
        return 1;
    }

    /// Load sparse matrix in COO format
    ptiAssert(ptiLoadSparseMatrix(&mtx, 1, fi) == 0);
    fclose(fi);
    ptiRandomValueVector(&(mtx.values));    // to better compare results
    ptiSparseMatrixStatus(&mtx, stdout);
    // ptiAssert(ptiDumpSparseMatrix(&mtx, 0, stdout) == 0);

    /* Convert to HiCOO */
    ptiNnzIndex max_nnzb = 0;
    ptiAssert(ptiSparseMatrixToHiCOO(&himtx, &max_nnzb, &mtx, sb_bits, sk_bits) == 0);
    ptiFreeSparseMatrix(&mtx);
    ptiSparseMatrixStatusHiCOO(&himtx, stdout);
    // ptiAssert(ptiDumpSparseMatrixHiCOO(&himtx, stdout) == 0);

    /// Initialize values for vectors x and y
    ptiNewValueVector(&x, himtx.ncols, himtx.ncols);
    ptiRandomValueVector(&x);
    ptiNewValueVector(&y, himtx.nrows, himtx.nrows);
    // ptiAssert(ptiDumpValueVector(&x, stdout) == 0);
    // ptiAssert(ptiDumpValueVector(&y, stdout) == 0);

    /* determine niters or num_kernel_dim to be parallelized */
    ptiIndex sk = (ptiIndex)pow(2, sk_bits);
    ptiIndex num_kernel_dim = (himtx.nrows + sk - 1) / sk;
    printf("num_kernel_dim: %u, himtx.nkiters / num_kernel_dim: %u\n", num_kernel_dim, himtx.nkiters/num_kernel_dim);
    if(num_kernel_dim <= NUM_CORES && himtx.nkiters / num_kernel_dim >= 20) {
        par_iters = 1;
    }

    /* Set zeros for temporary ybufs */
    char * bytestr;
    if(cuda_dev_id == -1 && par_iters == 1) {
        ybufs = (ptiValueVector *) malloc(nthreads * sizeof(ptiValueVector));
        for(int t=0; t<nthreads; ++t) {
            ptiNewValueVector(&ybufs[t], himtx.nrows, himtx.nrows);
            ptiConstantValueVector(&ybufs[t], 0);
        }
        ptiNnzIndex bytes = nthreads * himtx.nrows * sizeof(ptiValue);
        bytestr = ptiBytesString(bytes);
        printf("MATRIX BUFFER=%s\n\n", bytestr);
        free(bytestr);
    }

    // Warm-up
    if(cuda_dev_id == -2) {
        printf("Run ptiSparseMatrixMulVectorHiCOO:\n");
        ptiSparseMatrixMulVectorHiCOO(&y, &himtx, &x);
    } else if(cuda_dev_id == -1) {
        if(use_schedule == 1) {
            if(par_iters == 0) {
                printf("Run ptiOmpSparseMatrixMulVectorHiCOO_Schedule:\n");
                ptiOmpSparseMatrixMulVectorHiCOO_Schedule(&y, &himtx, &x);
            } else {
                printf("Run ptiOmpSparseMatrixMulVectorHiCOO_Schedule_Reduce:\n");
                ptiOmpSparseMatrixMulVectorHiCOO_Schedule_Reduce(&y, ybufs, &himtx, &x);
            }
        } else {
            printf("Run ptiOmpSparseMatrixMulVectorHiCOO:\n");
            ptiOmpSparseMatrixMulVectorHiCOO(&y, &himtx, &x);
        }
    }

    ptiStartTimer(timer);
    for(int i=0; i<niters; ++i) {
        if(cuda_dev_id == -2) {
            ptiSparseMatrixMulVectorHiCOO(&y, &himtx, &x);
        } else if(cuda_dev_id == -1) {
            if(use_schedule == 1) {
                if(par_iters == 0) {
                    ptiOmpSparseMatrixMulVectorHiCOO_Schedule(&y, &himtx, &x);
                } else {
                    ptiOmpSparseMatrixMulVectorHiCOO_Schedule_Reduce(&y, ybufs, &himtx, &x);
                }
            } else {
                ptiOmpSparseMatrixMulVectorHiCOO(&y, &himtx, &x);
            }
        }
    }
    ptiStopTimer(timer);
    printf("\n");
    double elapsed_time = ptiPrintAverageElapsedTime(timer, niters, "HiCOO-SpMV");
    ptiNnzIndex flops = 2 * himtx.nnz;
    ptiPrintGFLOPS(elapsed_time, flops, "HiCOO-SpMV");


    if(fo != NULL) {
        ptiAssert(ptiDumpValueVector(&y, fo) == 0);
        fclose(fo);
    }

    if(cuda_dev_id == -1 && par_iters == 1) {
        for(int t=0; t<nthreads; ++t) {
            ptiFreeValueVector(&ybufs[t]);
        }
        free(ybufs);
    }
    ptiFreeSparseMatrixHiCOO(&himtx);
    ptiFreeValueVector(&x);
    ptiFreeValueVector(&y);
    ptiFreeTimer(timer);

    return 0;
}
