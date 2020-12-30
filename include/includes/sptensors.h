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

#ifndef HIPARTI_SPTENSORS_H
#define HIPARTI_SPTENSORS_H

/* Sparse tensor */
int ptiNewSparseTensor(ptiSparseTensor *tsr, ptiIndex nmodes, const ptiIndex ndims[]);
int ptiCopySparseTensor(ptiSparseTensor *dest, const ptiSparseTensor *src, int const nt);
void ptiFreeSparseTensor(ptiSparseTensor *tsr);
double SparseTensorFrobeniusNormSquared(ptiSparseTensor const * const ptien);
int ptiLoadSparseTensor(ptiSparseTensor *tsr, ptiIndex start_index, char const * const fname);
int ptiDumpSparseTensor(const ptiSparseTensor *tsr, ptiIndex start_index, FILE *fp);
int ptiMatricize(ptiSparseTensor const * const X,
    ptiIndex const m,
    ptiSparseMatrix * const A,
    int const transpose);
void ptiGetBestModeOrder(
    ptiIndex * mode_order,
    ptiIndex const mode,
    ptiIndex const * ndims,
    ptiIndex const nmodes);
void ptiGetWorstModeOrder(
    ptiIndex * mode_order,
    ptiIndex const mode,
    ptiIndex const * ndims,
    ptiIndex const nmodes);
void ptiGetRandomShuffleElements(ptiSparseTensor *tsr);
void ptiGetRandomShuffledIndices(ptiSparseTensor *tsr, ptiIndex ** map_inds);
void ptiSparseTensorShuffleIndices(ptiSparseTensor *tsr, ptiIndex ** map_inds);
void ptiSparseTensorInvMap(ptiSparseTensor *tsr, ptiIndex ** in_map_inds);
void ptiSparseTensorShuffleModes(ptiSparseTensor *tsr, ptiIndex * mode_order);
int ptiSparseTensorSetIndices(
    ptiSparseTensor *ref,
    ptiIndex * mode_order,
    ptiIndex num_cmodes,
    ptiNnzIndexVector *fiberidx);
void ptiSparseTensorSortIndex(ptiSparseTensor *tsr, int force, int tk);
void ptiSparseTensorSortIndexAtMode(ptiSparseTensor *tsr, ptiIndex const mode, int force);
void ptiSparseTensorSortIndexCustomOrder(ptiSparseTensor *tsr, ptiIndex const *  mode_order, int force, int tk);
void ptiSparseTensorSortIndexMorton(
    ptiSparseTensor *tsr,
    int force,
    ptiNnzIndex begin,
    ptiNnzIndex end,
    ptiElementIndex sb_bits,
    int tk);
void ptiSparseTensorSortIndexExceptSingleModeRowBlock(
    ptiSparseTensor *tsr,
    int force,
    ptiNnzIndex begin,
    ptiNnzIndex end,
    ptiIndex * const mode_order,
    ptiElementIndex sk_bits,
    int tk);
void ptiSparseTensorSortIndexRowBlock(
    ptiSparseTensor *tsr,
    int force,
    ptiNnzIndex begin,
    ptiNnzIndex end,
    ptiElementIndex sk_bits,
    int tk);
void ptiSparseTensorSortIndexSingleMode(ptiSparseTensor *tsr, int force, ptiIndex mode, int tk);
void ptiSparseTensorSortIndexExceptSingleMode(ptiSparseTensor *tsr, int force, ptiIndex * mode_order, int tk);
void ptiSparseTensorSortIndexExceptSingleModeMorton(ptiSparseTensor *tsr, int force, ptiIndex * mode_order, ptiElementIndex sb_bits, int tk);
int ptiSparseTensorMixedOrder(
    ptiSparseTensor *tsr,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits,
    int const tk);
int ptiSparseTensorSortPartialIndex(
    ptiSparseTensor *tsr,
    ptiIndex const *  mode_order,
    const ptiElementIndex sb_bits,
    int const tk);
void ptiSparseTensorCalcIndexBounds(ptiIndex inds_low[], ptiIndex inds_high[], const ptiSparseTensor *tsr);
int pti_ComputeSliceSizes(
    ptiNnzIndex * slice_nnzs,
    ptiSparseTensor * const tsr,
    ptiIndex const mode);
void ptiSparseTensorStatus(ptiSparseTensor *tsr, FILE *fp);
double ptiSparseTensorDensity(ptiSparseTensor const * const tsr);

/* Renumbering */
void ptiIndexRenumber(ptiSparseTensor * tsr, ptiIndex ** newIndices, int renumber, ptiIndex iterations, ptiElementIndex sb_bits, int tk, int impl_num);
void orderit(ptiSparseTensor *tsr, ptiIndex **newIndices, int const renumber, ptiIndex const iterations);

/* Sparse tensor HiCOO */
int ptiNewSparseTensorHiCOO(
    ptiSparseTensorHiCOO *hitsr,
    const ptiIndex nmodes,
    const ptiIndex ndims[],
    const ptiNnzIndex nnz,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits,
    const ptiElementIndex sc_bits);
int ptiNewSparseTensorHiCOO_NoNnz(
    ptiSparseTensorHiCOO *hitsr,
    const ptiIndex nmodes,
    const ptiIndex ndims[],
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits,
    const ptiElementIndex sc_bits);
void ptiFreeSparseTensorHiCOO(ptiSparseTensorHiCOO *hitsr);
int ptiSparseTensorToHiCOO(
    ptiSparseTensorHiCOO *hitsr,
    ptiNnzIndex *max_nnzb,
    ptiSparseTensor *tsr,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits,
    const ptiElementIndex sc_bits,
    int const tk);
int ptiDumpSparseTensorHiCOO(ptiSparseTensorHiCOO * const hitsr, FILE *fp);
void ptiLoadShuffleFile(ptiSparseTensor *tsr, FILE *fs, ptiIndex ** map_inds);
void ptiSparseTensorStatusHiCOO(ptiSparseTensorHiCOO *hitsr, FILE *fp);
double SparseTensorFrobeniusNormSquaredHiCOO(ptiSparseTensorHiCOO const * const hitsr);
int ptiSetKernelPointers(
    ptiNnzIndexVector *kptr,
    ptiNnzIndexVector *knnzs,
    ptiSparseTensor *tsr,
    const ptiElementIndex sk_bits);


/* Sparse tensor unary operations */
int ptiSparseTensorMulScalar(ptiSparseTensor *X, ptiValue const a);
int ptiCudaSparseTensorMulScalar(ptiSparseTensor *X, ptiValue a);
int ptiSparseTensorDivScalar(ptiSparseTensor *X, ptiValue const a);

/* Sparse tensor binary operations */
int ptiSparseTensorAdd(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiSparseTensorSub(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiSparseTensorAddOMP(ptiSparseTensor *Y, ptiSparseTensor *X, int const nthreads);
int ptiSparseTensorSubOMP(ptiSparseTensor *Y, ptiSparseTensor *X, int const nthreads);

int ptiSparseTensorDotMul(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiSparseTensorDotMulEq(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiOmpSparseTensorDotMulEq(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiCudaSparseTensorDotMulEq(ptiSparseTensor *Z, const ptiSparseTensor *X, const ptiSparseTensor *Y);
int ptiSparseTensorDotDiv(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);

int ptiSparseTensorMulMatrix(ptiSemiSparseTensor *Y, ptiSparseTensor * const X, ptiMatrix *const U, ptiIndex mode);
int ptiOmpSparseTensorMulMatrix(ptiSemiSparseTensor *Y, ptiSparseTensor * const X, ptiMatrix * const U, ptiIndex mode);
int ptiCudaSparseTensorMulMatrix(ptiSemiSparseTensor *Y, ptiSparseTensor *X, const ptiMatrix *U, ptiIndex const mode);
int ptiCudaSparseTensorMulMatrixOneKernel(ptiSemiSparseTensor *Y, ptiSparseTensor *X, const ptiMatrix *U, ptiIndex const mode, ptiIndex const impl_num, ptiNnzIndex const smen_size);

int ptiSparseTensorMulVector(ptiSemiSparseTensor *Y, ptiSparseTensor * const X, ptiValueVector * const V, ptiIndex mode);

int ptiSparseTensorMulTensor(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor *const Y, ptiIndex num_cmodes, ptiIndex * cmodes_X, ptiIndex * cmodes_Y);

/**
 * Kronecker product
 */
int ptiSparseTensorKroneckerMul(ptiSparseTensor *Y, const ptiSparseTensor *A, const ptiSparseTensor *B);

/**
 * Khatri-Rao product
 */
int ptiSparseTensorKhatriRaoMul(ptiSparseTensor *Y, const ptiSparseTensor *A, const ptiSparseTensor *B);


/**
 * Matricized tensor times Khatri-Rao product.
 */
int ptiMTTKRP(
    ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode);
int ptiOmpMTTKRP(
    ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRP_Reduce(
    ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRP_Lock(
    ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    ptiMutexPool * lock_pool);
int ptiCudaMTTKRP(
    ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex * const mats_order,    // Correspond to the mode order of X.
    ptiIndex const mode,
    ptiIndex const impl_num);
int ptiCudaMTTKRPOneKernel(
    ptiSparseTensor const * const X,
    ptiMatrix ** const mats,     // mats[nmodes] as temporary space.
    ptiIndex * const mats_order,    // Correspond to the mode order of X.
    ptiIndex const mode,
    ptiIndex const impl_num);
int ptiCudaMTTKRPSM(
    ptiSparseTensor const * const X,
    ptiMatrix ** const mats,     // mats[nmodes] as temporary space.
    ptiIndex * const mats_order,    // Correspond to the mode order of X.
    ptiIndex const mode,
    ptiIndex const impl_num);
int ptiCudaMTTKRPDevice(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex rank,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats,
    ptiValue * dev_scratch);
int ptiSplittedMTTKRP(
    ptiSparseTensor const *const X,
    ptiMatrix *mats[],
    ptiIndex const mats_order[],
    ptiIndex const mode,
    ptiValueVector *scratch,
    ptiIndex const split_count[]
);

/* Coarse GPU */
int ptiCudaCoarseMTTKRP(
    ptiSparseTensor const * const X,
    ptiMatrix ** const mats,     // mats[nmodes] as temporary space.
    ptiIndexVector const * const mats_order,    // Correspond to the mode order of X.
    ptiIndex const mode);

/**
 * Matricized tensor times Khatri-Rao product for HiCOO tensors
 */
int ptiMTTKRPHiCOO(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode);
int ptiMTTKRPHiCOO_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode);
int ptiOmpMTTKRPHiCOO(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb);
int ptiOmpMTTKRPHiCOO_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb);
int ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb,
    int balanced);
int ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb,
    int balanced);
int ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce_Two(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb);
int ptiCudaMTTKRPHiCOO(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix ** const mats,     // mats[nmodes] as temporary space.
    ptiIndex * const mats_order,    // Correspond to the mode order of X.
    ptiIndex const mode,
    ptiNnzIndex const max_nnzb,
    int const impl_num);
int ptiMTTKRPKernelHiCOO(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiNnzIndex max_nnzb,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sc_bits,
    const ptiIndex blength,
    const int impl_num,
    const ptiNnzIndex kptr_begin,
    const ptiNnzIndex kptr_end,
    ptiIndex * const dev_ndims,
    ptiNnzIndex * const dev_cptr,
    ptiNnzIndex * const dev_bptr,
    ptiBlockIndex ** const dev_binds,
    ptiElementIndex ** const dev_einds,
    ptiValue * const dev_values,
    ptiIndex * const dev_mats_order,
    ptiValue ** const dev_mats);
int ptiTTMHiCOO_MatrixTiling(
    ptiSparseTensorHiCOO * const Y,
    ptiSparseTensorHiCOO const * const X,
    ptiRankMatrix * U,     // mats[nmodes] as temporary space.
    ptiIndex const mode);


#endif
