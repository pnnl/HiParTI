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
    ptiNnzIndex * bptr; // non-zero block pointers
    ptiElementIndex sb_bits = 7;    // block size
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    /* OpenMP */
    int cuda_dev_id = -2;   // Run sequential code by default
    int nthreads = 1;   // get from OMP_NUM_THREADS environment

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"bs", optional_argument, 0, 'b'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:b:d:", long_options, &option_index);
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
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        default:
            abort();
        }
    }
    printf("sb: %ld\n", (long int)pow(2,sb_bits));
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
        printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
        printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
        printf("\n");
        return 1;
    }

    /// Load sparse matrix in COO format
    ptiAssert(ptiLoadSparseMatrix(&mtx, 1, fi) == 0);
    fclose(fi);
    ptiRandomValueVector(&(mtx.values));    // to better compare results
    ptiSparseMatrixStatus(&mtx, stdout);
    // printf("Input COO Matrix\n");
    // ptiAssert(ptiDumpSparseMatrix(&mtx, 0, stdout) == 0);

    ptiNnzIndex max_nnzb = 0;
    ptiStartTimer(timer);
    ptiNewSparseMatrixHiCOO(&himtx, mtx.nrows, mtx.ncols, mtx.nnz, sb_bits, sb_bits);

    /* Sort blocks */
    ptiStartTimer(timer);
    // Natural block sorting
    ptiSparseMatrixSortIndexRowBlock(&mtx, 1, 0, mtx.nnz, sb_bits);  // OMP-Parallelized inside
    // Morton-order block sorting
    // ptiSparseMatrixSortIndexMorton(&mtx, 1, 0, mtx.nnz, sb_bits);  // OMP-Parallelized inside
    ptiAppendNnzIndexVector(&himtx.kptr, 0);    ptiAppendNnzIndexVector(&himtx.kptr, mtx.nnz); // To ease the use of HiCOO code
    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "Sort");
    // printf("Sorted COO Matrix\n");
    // ptiAssert(ptiDumpSparseMatrix(&mtx, 0, stdout) == 0);

    /* Partitioning */
    ptiStartTimer(timer);
    ptiSparseMatrixPartition(&himtx, &max_nnzb, &mtx, sb_bits);   // Create a HiCOO matrix underneath
    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "Partition");
    bptr = himtx.bptr.data; // Extract block pointers from HiCOO matrix struct
    // printf("Block pointers:\n");
    // ptiAssert(ptiDumpNnzIndexArray(bptr, himtx.bptr.len, stdout) == 0);

    ptiFreeSparseMatrixHiCOO(&himtx);
    ptiFreeSparseMatrix(&mtx);
    ptiFreeTimer(timer);

    return 0;
}
