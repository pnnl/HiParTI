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
    ptiSparseMatrixHiCOO hispA;
    ptiMatrix B, C;
    ptiElementIndex sb_bits = 7;    // 2^7 by default
    ptiIndex R = 16;
    int niters = 5;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    /* OpenMP */
    int cuda_dev_id = -2;
    int nthreads = 1;   // get from OMP_NUM_THREADS environment
    ptiElementIndex sk_bits = sb_bits;
    int use_schedule = 0; // privatization or not
    int par_iters = 0;  // determine in the code
    ptiMatrix * Cbufs;
    double min_time = 100000.0;
    ptiElementIndex min_time_sk_bits;

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"bs", optional_argument, 0, 'b'},
        {"R", optional_argument, 0, 'r'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {"use-schedule", optional_argument, 0, 'u'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:b:r:d:u:", long_options, &option_index);
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
        case 'b':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'r':
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &R);
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
    printf("B ncols: %d\n", R);
    printf("niters: %d\n", niters);
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    if(cuda_dev_id == -1) {
        printf("use_schedule: %d\n", use_schedule);
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
        printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
        printf("         -R RANK\n");
        printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
        printf("         -u use_schedule, --ur=use_schedule\n");
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

    /* Set max_sk */

    ptiIndex min_ndims = (spA.nrows > spA.ncols) ? spA.ncols : spA.nrows;
    ptiElementIndex max_bits = (ptiElementIndex)(log2(min_ndims));
    if(max_bits < sb_bits) {
        sb_bits = max_bits;
        sk_bits = max_bits;
    }
    if(cuda_dev_id == -2) max_bits = sk_bits;   // only run one case for sequential code
    printf("sb: %ld\n", (long int)pow(2,sb_bits));
#ifdef HIPARTI_USE_OPENMP
    printf("max_bits: %d (%ld)\n", (int)(max_bits), (long int)pow(2,max_bits));
    for(ptiElementIndex sk_bits = sb_bits; sk_bits <= max_bits; sk_bits += 1) {
        printf("sk: %ld\n", (long int)pow(2,sk_bits));
#else
    {
#endif

        /* Convert to HiCOO */
        ptiNnzIndex max_nnzb = 0;
        ptiAssert(ptiSparseMatrixToHiCOO(&hispA, &max_nnzb, &spA, sb_bits, sk_bits) == 0);   // TODO
        ptiSparseMatrixStatusHiCOO(&hispA, stdout);
        // ptiAssert(ptiDumpSparseMatrixHiCOO(&hispA, stdout) == 0);

        /* determine niters or num_kernel_dim to be parallelized */
    #ifdef HIPARTI_USE_OPENMP
        ptiIndex sk = (ptiIndex)pow(2, hispA.sk_bits);
        ptiIndex num_kernel_dim = (hispA.nrows + sk - 1) / sk;
        printf("num_kernel_dim: %u, hispA.nkiters / num_kernel_dim: %u\n", num_kernel_dim, hispA.nkiters/num_kernel_dim);
        if(num_kernel_dim <= NUM_CORES && hispA.nkiters / num_kernel_dim >= 20) {
            par_iters = 1;
        }

        /* Set zeros for temporary CBufs */
        char * bytestr;
        if(cuda_dev_id == -1 && par_iters == 1) {
            Cbufs = (ptiMatrix *)malloc(nthreads * sizeof(ptiMatrix));
            for(int t=0; t<nthreads; ++t) {
                ptiAssert(ptiNewMatrix(&Cbufs[t], hispA.nrows, R) == 0);
                ptiAssert(ptiConstantMatrix(&Cbufs[t], 0) == 0);
            }
            ptiNnzIndex bytes = nthreads * hispA.nrows * R * sizeof(ptiValue);
            bytestr = ptiBytesString(bytes);
            printf("MATRIX BUFFER=%s\n\n", bytestr);
            free(bytestr);
        }
    #endif

        // Warm-up
        if(cuda_dev_id == -2) {
            printf("Run ptiSparseMatrixMulMatrixHiCOO:\n");
            ptiSparseMatrixMulMatrixHiCOO(&C, &hispA, &B);
        } else if(cuda_dev_id == -1) {
            if(use_schedule == 1) {
                if(par_iters == 0) {
                    printf("Run ptiOmpSparseMatrixMulMatrixHiCOO_Schedule:\n");
                    ptiOmpSparseMatrixMulMatrixHiCOO_Schedule(&C, &hispA, &B);
                } else {
                    printf("Run ptiOmpSparseMatrixMulMatrixHiCOO_Schedule_Reduce:\n");
                    ptiOmpSparseMatrixMulMatrixHiCOO_Schedule_Reduce(&C, Cbufs, &hispA, &B);
                }
            } else {
                printf("Run ptiOmpSparseMatrixMulMatrixHiCOO:\n");
                ptiOmpSparseMatrixMulMatrixHiCOO(&C, &hispA, &B);
            }
        }

        ptiStartTimer(timer);
        for(int i=0; i<niters; ++i) {
            if(cuda_dev_id == -2) {
                ptiSparseMatrixMulMatrixHiCOO(&C, &hispA, &B);
            } else if(cuda_dev_id == -1) {
                if(use_schedule == 1) {
                    if(par_iters == 0) {
                        ptiOmpSparseMatrixMulMatrixHiCOO_Schedule(&C, &hispA, &B);
                    } else {
                        ptiOmpSparseMatrixMulMatrixHiCOO_Schedule_Reduce(&C, Cbufs, &hispA, &B);
                    }
                } else {
                    ptiOmpSparseMatrixMulMatrixHiCOO(&C, &hispA, &B);
                }
            }
        }
        ptiStopTimer(timer);
        printf("\n");
        double elapsed_time = ptiPrintAverageElapsedTime(timer, niters, "HiCOO-SpMM");
        fflush(stdout);

        /* Get the best perf */
        if (min_time > elapsed_time) {
            min_time = elapsed_time;
            min_time_sk_bits = sk_bits;
        }

        ptiFreeSparseMatrixHiCOO(&hispA);
    }   // End Loop of sks

    ptiNnzIndex flops = 2 * spA.nnz * R;
    ptiPrintGFLOPS(min_time, flops, "HiCOO-SpMM");
    if(cuda_dev_id == -1)
        printf("min_time_sk_bits: %u (%ld) \n", (unsigned)min_time_sk_bits, (long int)pow(2,min_time_sk_bits));


    if(fo != NULL) {
        ptiAssert(ptiDumpMatrix(&C, fo) == 0);
        fclose(fo);
    }

#ifdef HIPARTI_USE_OPENMP
    if(cuda_dev_id == -1 && par_iters == 1) {
        for(int t=0; t<nthreads; ++t) {
            ptiFreeMatrix(&Cbufs[t]);
        }
        free(Cbufs);
    }
#endif

    ptiFreeSparseMatrix(&spA);
    ptiFreeMatrix(&B);
    ptiFreeMatrix(&C);
    ptiFreeTimer(timer);

    return 0;
}
