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

#ifndef HIPARTI_MATRICES_H
#define HIPARTI_MATRICES_H

/* Dense matrix */
static inline ptiNnzIndex ptiGetMatrixLength(const ptiMatrix *mtx) {
    return mtx->nrows * mtx->stride;
}
int ptiNewMatrix(ptiMatrix *mtx, ptiIndex const nrows, ptiIndex const ncols);
int ptiRandomizeMatrix(ptiMatrix *mtx);
int ptiIdentityMatrix(ptiMatrix *mtx);
int ptiConstantMatrix(ptiMatrix * const mtx, ptiValue const val);
void ptiMatrixInverseShuffleIndices(ptiMatrix *mtx, ptiIndex * mode_map_inds);
int ptiCopyMatrix(ptiMatrix *dest, const ptiMatrix *src);
int ptiAppendMatrix(ptiMatrix *mtx, const ptiValue values[]);
int ptiResizeMatrix(ptiMatrix *mtx, ptiIndex const new_nrows);
void ptiFreeMatrix(ptiMatrix *mtx);
int ptiDumpMatrix(ptiMatrix *mtx, FILE *fp);

/* Dense matrix operations */
int ptiMatrixDotMul(ptiMatrix const * A, ptiMatrix const * B, ptiMatrix const * C);
int ptiMatrixDotMulSeq(ptiIndex const mode, ptiIndex const nmodes, ptiMatrix ** mats);
int ptiCudaMatrixDotMulSeq(
    ptiIndex const mode,
    ptiIndex const nmodes,
    ptiIndex const rank,
    ptiIndex const stride,
    ptiValue ** dev_ata);
int ptiMatrixDotMulSeqCol(ptiIndex const mode, ptiIndex const nmodes, ptiMatrix ** mats);
int ptiMatrixDotMulSeqTriangle(ptiIndex const mode, ptiIndex const nmodes, ptiMatrix ** mats);
int ptiMatrix2Norm(ptiMatrix * const A, ptiValue * const lambda);
int ptiCudaMatrix2Norm(
    ptiIndex const nrows,
    ptiIndex const ncols,
    ptiIndex const stride,
    ptiValue * const dev_vals,
    ptiValue * const dev_lambda);
int ptiMatrixMaxNorm(ptiMatrix * const A, ptiValue * const lambda);
void GetFinalLambda(
  ptiIndex const rank,
  ptiIndex const nmodes,
  ptiMatrix ** mats,
  ptiValue * const lambda);
int ptiMatrixSolveNormals(
  ptiIndex const mode,
  ptiIndex const nmodes,
  ptiMatrix ** aTa,
  ptiMatrix * rhs);
int ptiSparseTensorToMatrix(ptiMatrix *dest, const ptiSparseTensor *src);

/* Dense Rank matrix, ncols = small rank (<= 256) */
int ptiNewRankMatrix(ptiRankMatrix *mtx, ptiIndex const nrows, ptiElementIndex const ncols);
int ptiRandomizeRankMatrix(ptiRankMatrix *mtx, ptiIndex const nrows, ptiElementIndex const ncols);
int ptiConstantRankMatrix(ptiRankMatrix *mtx, ptiValue const val);
void ptiRankMatrixInverseShuffleIndices(ptiRankMatrix *mtx, ptiIndex * mode_map_inds);
void ptiFreeRankMatrix(ptiRankMatrix *mtx);
int ptiDumpRankMatrix(ptiRankMatrix *mtx, FILE *fp);

/* Dense rank matrix operations */
int ptiRankMatrixDotMulSeqTriangle(ptiIndex const mode, ptiIndex const nmodes, ptiRankMatrix ** mats);
int ptiRankMatrix2Norm(ptiRankMatrix * const A, ptiValue * const lambda);
int ptiRankMatrixMaxNorm(ptiRankMatrix * const A, ptiValue * const lambda);
void GetRankFinalLambda(
  ptiElementIndex const rank,
  ptiIndex const nmodes,
  ptiRankMatrix ** mats,
  ptiValue * const lambda);
int ptiRankMatrixSolveNormals(
  ptiIndex const mode,
  ptiIndex const nmodes,
  ptiRankMatrix ** aTa,
  ptiRankMatrix * rhs);

#endif