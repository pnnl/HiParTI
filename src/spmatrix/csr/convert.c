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
#include <stdlib.h>
#include <string.h>

int ptiSparseMatrixToCSR (ptiSparseMatrixCSR * csrmtx, ptiSparseMatrix * mtx)
{
  ptiNnzIndex nnz = mtx->nnz;
  ptiIndex nrows = mtx->nrows;

  ptiNewSparseMatrixCSR(csrmtx, nrows, mtx->ncols, nnz);

  for (ptiIndex i = 0; i < nrows + 1; i++)
      csrmtx->rowptr.data[i] = 0;

  for (ptiNnzIndex i = 0; i < nnz; i++)
      csrmtx->rowptr.data[ mtx->rowind.data[i] ] ++;

  // cumsum the nnz per row to get rowptr
  for(ptiIndex i = 0, cumsum = 0; i < nrows; i++) {
      ptiNnzIndex temp = csrmtx->rowptr.data[i];
      csrmtx->rowptr.data[i] = cumsum;
      cumsum += temp;
  }
  csrmtx->rowptr.data[nrows] = nnz;

  for(ptiNnzIndex i = 0; i < nnz; i++) {
      ptiIndex row  = mtx->rowind.data[i];
      ptiNnzIndex dest = csrmtx->rowptr.data[row];

      csrmtx->colind.data[dest] = mtx->colind.data[i];
      csrmtx->values.data[dest] = mtx->values.data[i];

      csrmtx->rowptr.data[row] ++;
  }

  // Recover rowptr
  for(ptiIndex i = 0, last = 0; i <= nrows; i++) {
      ptiNnzIndex temp = csrmtx->rowptr.data[i];
      csrmtx->rowptr.data[i]  = last;
      last   = temp;
  }

  return 0;
}