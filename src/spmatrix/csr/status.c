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


void ptiSparseMatrixStatusCSR(ptiSparseMatrixCSR *csrmtx, FILE *fp)
{
  fprintf(fp, "CSR Sparse Matrix information (use ptiIndex, ptiValue))---------\n");
  fprintf(fp, " DIMS=%"HIPARTI_PRI_INDEX "x%"HIPARTI_PRI_INDEX "\n", csrmtx->nrows, csrmtx->ncols);
  fprintf(fp, " NNZ=%"HIPARTI_PRI_NNZ_INDEX, csrmtx->nnz);

  char * bytestr = ptiBytesString(csrmtx->nnz * (sizeof(ptiIndex) + sizeof(ptiValue)) + (csrmtx->nrows + 1) * sizeof(ptiNnzIndex));
  fprintf(fp, " CSR-STORAGE=%s\n", bytestr);
  fprintf(fp, "\n");
  free(bytestr);
}
