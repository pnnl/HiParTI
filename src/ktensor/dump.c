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

#include <HiParTI.h>
#include <stdlib.h>
#include <string.h>

int ptiDumpKruskalTensor(ptiKruskalTensor *ktsr, FILE *fp)
{
    int iores;
    ptiIndex mode;

    iores = fprintf(fp, "nmodes: %"HIPARTI_PRI_INDEX ", rank: %"HIPARTI_PRI_INDEX "\n", ktsr->nmodes, ktsr->rank);
    pti_CheckOSError(iores < 0, "KruskalTns Dump");
    for(mode = 0; mode < ktsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            pti_CheckOSError(iores < 0, "KruskalTns Dump");
        }
        iores = fprintf(fp, "%"HIPARTI_PRI_INDEX, ktsr->ndims[mode]);
        pti_CheckOSError(iores < 0, "KruskalTns Dump");
    }
    fputs("\n", fp);

    iores = fprintf(fp, "fit: %lf\n", ktsr->fit);
    fprintf(fp, "lambda:\n");    
    for(ptiIndex r = 0; r < ktsr->rank; ++r) {
        iores = fprintf(fp, "%"HIPARTI_PRI_VALUE " ", ktsr->lambda[r]);
        // pti_CheckOSError(iores != 0, "KruskalTns Dump");
    }
    fprintf(fp, "\n");

    fprintf(fp, "Factor matrices:\n");
    for(mode=0; mode < ktsr->nmodes; ++mode) {
        iores = ptiDumpMatrix(ktsr->factors[mode], fp);
        pti_CheckOSError(iores != 0, "KruskalTns Dump");
    }
    return 0;
}


int ptiDumpRankKruskalTensor(ptiRankKruskalTensor *ktsr, FILE *fp)
{
    int iores;
    ptiIndex mode;

    iores = fprintf(fp, "nmodes: %"HIPARTI_PRI_INDEX ", rank: %"HIPARTI_PRI_ELEMENT_INDEX "\n", ktsr->nmodes, ktsr->rank);
    pti_CheckOSError(iores < 0, "RankKruskalTns Dump");

    for(mode = 0; mode < ktsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            pti_CheckOSError(iores < 0, "RankKruskalTns Dump");
        }
        iores = fprintf(fp, "%"HIPARTI_PRI_INDEX, ktsr->ndims[mode]);
        pti_CheckOSError(iores < 0, "RankKruskalTns Dump");
    }
    fputs("\n", fp);

    iores = fprintf(fp, "fit: %lf\n", ktsr->fit);
    fprintf(fp, "lambda:\n");    
    for(ptiElementIndex r = 0; r < ktsr->rank; ++r) {
        iores = fprintf(fp, "%"HIPARTI_PRI_VALUE " ", ktsr->lambda[r]);
        // pti_CheckOSError(iores != 0, "RankKruskalTns Dump");
    }
    fprintf(fp, "\n");

    fprintf(fp, "Factor matrices:\n");
    for(mode=0; mode < ktsr->nmodes; ++mode) {
        iores = ptiDumpRankMatrix(ktsr->factors[mode], fp);
        pti_CheckOSError(iores != 0, "RankKruskalTns Dump");
    }
    return 0;
}