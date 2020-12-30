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
#include <HiParTI.h>

int main(int argc, char *argv[]) {
    FILE *fi, *fo;
    ptiSparseMatrix mtx;

    if(argc != 3) {
        printf("Usage: %s input output\n\n", argv[0]);
        return 1;
    }

    fi = fopen(argv[1], "r");
    ptiAssert(fi != NULL);
    ptiAssert(ptiLoadSparseMatrix(&mtx, 1, fi) == 0);
    fclose(fi);

    fo = fopen(argv[2], "w");
    ptiAssert(fo != NULL);
    ptiAssert(ptiDumpSparseMatrix(&mtx, 1, fo) == 0);
    fclose(fo);

    ptiFreeSparseMatrix(&mtx);

    return 0;
}
