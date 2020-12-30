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
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/**
 * Dum a dense matrix to file
 *
 * @param mtx   a valid pointer to a ptiMatrix variable
 * @param fp a file pointer
 *
 */
int ptiDumpMatrix(ptiMatrix *mtx, FILE *fp) {
    int iores;
    ptiIndex nrows = mtx->nrows;
    ptiIndex ncols = mtx->ncols;
    ptiIndex stride = mtx->stride;
    iores = fprintf(fp, "%"HIPARTI_PRI_INDEX " x %"HIPARTI_PRI_INDEX " matrix\n", nrows, ncols);
    pti_CheckOSError(iores < 0, "Mtx Dump");
    for(ptiIndex i=0; i < nrows; ++i) {
      for(ptiIndex j=0; j < ncols; ++j) {
          iores = fprintf(fp, "%.2"HIPARTI_PRI_VALUE "\t", mtx->values[i * stride + j]);
          pti_CheckOSError(iores < 0, "Mtx Dump");
      }
      iores = fprintf(fp, "\n");
    }
    iores = fprintf(fp, "\n");
    return 0;
}


/**
 * Dum a dense rank matrix to file
 *
 * @param mtx   a valid pointer to a ptiMatrix variable
 * @param fp a file pointer
 *
 */
int ptiDumpRankMatrix(ptiRankMatrix *mtx, FILE *fp) {
    int iores;
    ptiIndex nrows = mtx->nrows;
    ptiElementIndex ncols = mtx->ncols;
    ptiElementIndex stride = mtx->stride;
    iores = fprintf(fp, "%"HIPARTI_PRI_INDEX " x %"HIPARTI_PRI_ELEMENT_INDEX " matrix\n", nrows, ncols);
    pti_CheckOSError(iores < 0, "RankMtx Dump");
    for(ptiIndex i=0; i < nrows; ++i) {
      for(ptiElementIndex j=0; j < ncols; ++j) {
          iores = fprintf(fp, "%.2"HIPARTI_PRI_VALUE "\t", mtx->values[i * stride + j]);
          pti_CheckOSError(iores < 0, "RankMtx Dump");
      }
      iores = fprintf(fp, "\n");
    }
    iores = fprintf(fp, "\n");
    return 0;
}
