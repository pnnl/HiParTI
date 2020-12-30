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

#include <HiParTI.h>
#include <math.h>


static double ptiSparseMatrixDensity(ptiSparseMatrix const * const mtx)
{
  double density = (double)mtx->nnz / ((double)mtx->nrows * mtx->ncols);
  return density;
}


void ptiSparseMatrixStatus(ptiSparseMatrix *mtx, FILE *fp)
{
  fprintf(fp, "COO Sparse Matrix information (use ptiIndex, ptiValue))---------\n");
  fprintf(fp, " DIMS=%"HIPARTI_PRI_INDEX "x%"HIPARTI_PRI_INDEX "\n", mtx->nrows, mtx->ncols);
  fprintf(fp, " NNZ=%"HIPARTI_PRI_NNZ_INDEX, mtx->nnz);
  fprintf(fp, " DENSITY=%e\n" , ptiSparseMatrixDensity(mtx));

  fprintf(fp, " Average row length: %.2lf\n", (double)mtx->nnz / mtx->nrows);
  fprintf(fp, " Average column length: %.2lf\n", (double)mtx->nnz / mtx->ncols);

  char * bytestr = ptiBytesString(mtx->nnz * (sizeof(ptiIndex) * 2 + sizeof(ptiValue)));
  fprintf(fp, " COO-STORAGE=%s\n", bytestr);
  fprintf(fp, "\n");
  free(bytestr);
}
