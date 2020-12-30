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

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:", long_options, &option_index);
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
        default:
            abort();
        }
    }

    if(optind > argc || argc < 3) {
        printf("Usage: %s\n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("\n");
        return 1;
    }

    ptiAssert(ptiLoadSparseMatrix(&mtx, 1, fi) == 0);
    fclose(fi);
    ptiSparseMatrixStatus(&mtx, stdout);
    // ptiAssert(ptiDumpSparseMatrix(&mtx, 0, stdout) == 0);

    ptiAssert(ptiSparseMatrixToCSR(&csrmtx, &mtx) == 0);
    ptiFreeSparseMatrix(&mtx);
    ptiSparseMatrixStatusCSR(&csrmtx, stdout);
    if(fo != NULL) {
        ptiAssert(ptiDumpSparseMatrixCSR(&csrmtx, fo) == 0);
        fclose(fo);
    }

    ptiFreeSparseMatrixCSR(&csrmtx);

    return 0;
}
