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

#ifndef HIPARTI_SPMATRIX_H
#define HIPARTI_SPMATRIX_H


/* Sparse matrix */
int ptiNewSparseMatrix(ptiSparseMatrix *mtx, ptiIndex nrows, ptiIndex ncols, ptiIndex nnz);
int ptiCopySparseMatrix(ptiSparseMatrix *dest, const ptiSparseMatrix *src, int const nt);
void ptiSparseMatrixShuffleIndices(ptiSparseMatrix *mtx, ptiIndex ** map_inds);
void ptiFreeSparseMatrix(ptiSparseMatrix *mtx);

int ptiLoadSparseMatrix(ptiSparseMatrix *mtx, ptiIndex start_index, FILE *fid);
int ptiDumpSparseMatrix(const ptiSparseMatrix *mtx, ptiIndex start_index, FILE *fp);

void ptiSparseMatrixStatus(ptiSparseMatrix *mtx, FILE *fp);

/* Sparse matrix operations */
int ptiSparseMatrixMulVector(ptiValueVector * y, const ptiSparseMatrix *mtx, ptiValueVector * x);
int ptiSparseMatrixMulMatrix(ptiMatrix * C, const ptiSparseMatrix *spA, ptiMatrix * B);
#ifdef HIPARTI_USE_OPENMP
int ptiMakeVectorBuff(ptiValueVector * ybufs, ptiIndex nrows);
int ptiFreeVecBuff(ptiValueVector * ybufs);
int ptiOmpSparseMatrixMulVector(ptiValueVector * y, const ptiSparseMatrix *mtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVectorReduce(ptiValueVector * y, const ptiSparseMatrix *mtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVector_Reduce(ptiValueVector * y, ptiValueVector * ybufs, const ptiSparseMatrix *mtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulMatrix(ptiMatrix * C, const ptiSparseMatrix *spA, ptiMatrix * B);
int ptiOmpSparseMatrixMulMatrix_Reduce(ptiMatrix * C, ptiMatrix * Cbufs, const ptiSparseMatrix *spA, ptiMatrix * B);
#endif

/* HiCOO */
int ptiNewSparseMatrixHiCOO(
    ptiSparseMatrixHiCOO *himtx,
    const ptiIndex nrows,
    const ptiIndex ncols,
    const ptiNnzIndex nnz,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits);
void ptiFreeSparseMatrixHiCOO(ptiSparseMatrixHiCOO *himtx);
int ptiDumpSparseMatrixHiCOO(ptiSparseMatrixHiCOO * const himtx, FILE *fp);
void ptiSparseMatrixStatusHiCOO(ptiSparseMatrixHiCOO *himtx, FILE *fp);
int ptiSparseMatrixToHiCOO(
    ptiSparseMatrixHiCOO *himtx,
    ptiNnzIndex *max_nnzb,
    ptiSparseMatrix *mtx,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits);
int ptiSparseMatrixPartition(
    ptiSparseMatrixHiCOO *himtx,
    ptiNnzIndex *max_nnzb,
    ptiSparseMatrix *mtx,
    const ptiElementIndex sb_bits);
void ptiSparseMatrixSortIndexMorton(
    ptiSparseMatrix *mtx,
    int force,
    ptiNnzIndex begin,
    ptiNnzIndex end,
    ptiElementIndex sb_bits);
void ptiSparseMatrixSortIndexSingleMode(ptiSparseMatrix *mtx, int force, ptiIndex mode, int tk);
void ptiSparseMatrixSortIndexRowBlock(
    ptiSparseMatrix *mtx,
    int force,
    ptiNnzIndex begin,
    ptiNnzIndex end,
    ptiElementIndex sk_bits);
int pti_SparseMatrixCompareIndicesSingleMode(ptiSparseMatrix * const mtx1, ptiNnzIndex loc1, ptiSparseMatrix * const mtx2, ptiNnzIndex loc2, ptiIndex const mode);

/* HiCOO operations */
int ptiSparseMatrixMulVectorHiCOO(ptiValueVector * y, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x);
int ptiSparseMatrixMulMatrixHiCOO(ptiMatrix * C, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B);
#ifdef HIPARTI_USE_OPENMP
int ptiOmpSparseMatrixMulVectorHiCOO(ptiValueVector * y, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVectorHiCOO_Schedule(ptiValueVector * y, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVectorHiCOOReduce(ptiValueVector * y, const ptiSparseMatrixHiCOO *mtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVectorHiCOO_Schedule_Reduce(ptiValueVector * y, ptiValueVector * ybufs, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulMatrixHiCOO(ptiMatrix * C, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B);
int ptiOmpSparseMatrixMulMatrixHiCOO_Schedule(ptiMatrix * C, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B);
int ptiOmpSparseMatrixMulMatrixHiCOO_Schedule_Reduce(ptiMatrix * C, ptiMatrix * Cbufs, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B);
#endif

/* Reordering */
void ptiIndexRelabel(ptiSparseMatrix * mtx, ptiIndex ** newIndices, int renumber, ptiIndex iterations, int tk);
void ptiGetRandomShuffledIndicesMat(ptiSparseMatrix *mtx, ptiIndex ** map_inds);


/* CSR */
int ptiNewSparseMatrixCSR(ptiSparseMatrixCSR *csrmtx, ptiIndex nrows, ptiIndex ncols, ptiIndex nnz);
void ptiFreeSparseMatrixCSR(ptiSparseMatrixCSR *csrmtx);
int ptiDumpSparseMatrixCSR(ptiSparseMatrixCSR * const csrmtx, FILE *fp);
void ptiSparseMatrixStatusCSR(ptiSparseMatrixCSR *csrmtx, FILE *fp);
int ptiSparseMatrixToCSR (ptiSparseMatrixCSR * csrmtx, ptiSparseMatrix * mtx);

/* CSR operations */
int ptiSparseMatrixMulVectorCSR(ptiValueVector * y, ptiSparseMatrixCSR *csrmtx, ptiValueVector * x);
int ptiSparseMatrixMulMatrixCSR(ptiMatrix * C, ptiSparseMatrixCSR *csrmtx, ptiMatrix * B);
#ifdef HIPARTI_USE_OPENMP
int ptiOmpSparseMatrixMulVectorCSR(ptiValueVector * y, ptiSparseMatrixCSR *csrmtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVectorCSRReduce(ptiValueVector * y, const ptiSparseMatrixCSR *mtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVectorCSR_Reduce(ptiValueVector *y, ptiValueVector * ybufs, const ptiSparseMatrixCSR *csrmtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulMatrixCSR(ptiMatrix * C, ptiSparseMatrixCSR *csrmtx, ptiMatrix * B);
#endif

#endif