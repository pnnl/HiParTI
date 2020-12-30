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
#include "../src/sptensor/sptensor.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    char ifname[1000];
    FILE *fo = NULL;
    ptiSparseTensor tsr;
    ptiSparseTensorHiCOO hitsr;
    ptiElementIndex sb_bits;
    ptiElementIndex sk_bits;
    ptiElementIndex sc_bits;

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
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:b:k:c:", long_options, &option_index);
        if(c == -1) {
            break;
        }

        switch(c) {
        case 'i':
            strcpy(ifname, optarg);
            printf("Input file: %s\n", ifname); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            ptiAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
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
        default:
            abort();
        }
    }

    ptiAssert(ptiLoadSparseTensor(&tsr, 1, ifname) == 0);
    ptiSparseTensorStatus(&tsr, stdout);
    // ptiAssert(ptiDumpSparseTensor(&tsr, 0, stdout) == 0);

    ptiNnzIndex max_nnzb = 0;
    ptiAssert(ptiSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, sc_bits, 1) == 0);
    ptiFreeSparseTensor(&tsr);
    ptiSparseTensorStatusHiCOO(&hitsr, stdout);

    if (fo != NULL) {
        ptiAssert(ptiDumpSparseTensorHiCOO(&hitsr, fo) == 0);
        fclose(fo);
    }

    ptiFreeSparseTensorHiCOO(&hitsr);

    return 0;
}
