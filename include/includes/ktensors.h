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

#ifndef HIPARTI_KTENSORS_H
#define HIPARTI_KTENSORS_H

/* Kruskal tensor */
int ptiNewKruskalTensor(ptiKruskalTensor *ktsr, ptiIndex nmodes, const ptiIndex ndims[], ptiIndex rank);
void ptiKruskalTensorInverseShuffleIndices(ptiKruskalTensor * ktsr, ptiIndex ** map_inds);
void ptiFreeKruskalTensor(ptiKruskalTensor *ktsr);
int ptiDumpKruskalTensor(ptiKruskalTensor *ktsr, FILE *fp);
double KruskalTensorFit(
  ptiSparseTensor const * const ptien,
  ptiValue const * const __restrict lambda,
  ptiMatrix ** mats,
  ptiMatrix ** ata);
double KruskalTensorFrobeniusNormSquared(
  ptiIndex const nmodes,
  ptiValue const * const __restrict lambda,
  ptiMatrix ** ata);
double SparseKruskalTensorInnerProduct(
  ptiIndex const nmodes,
  ptiValue const * const __restrict lambda,
  ptiMatrix ** mats);


/* Rank Kruskal tensor, ncols = small rank (<= 256)  */
int ptiNewRankKruskalTensor(ptiRankKruskalTensor *ktsr, ptiIndex nmodes, const ptiIndex ndims[], ptiElementIndex rank);
void ptiRankKruskalTensorInverseShuffleIndices(ptiRankKruskalTensor * ktsr, ptiIndex ** map_inds);
void ptiFreeRankKruskalTensor(ptiRankKruskalTensor *ktsr);
int ptiDumpRankKruskalTensor(ptiRankKruskalTensor *ktsr, FILE *fp);
double KruskalTensorFitHiCOO(
  ptiSparseTensorHiCOO const * const hitsr,
  ptiValue const * const __restrict lambda,
  ptiRankMatrix ** mats,
  ptiRankMatrix ** ata);
double KruskalTensorFrobeniusNormSquaredRank(
  ptiIndex const nmodes,
  ptiValue const * const __restrict lambda,
  ptiRankMatrix ** ata);
double SparseKruskalTensorInnerProductRank(
  ptiIndex const nmodes,
  ptiValue const * const __restrict lambda,
  ptiRankMatrix ** mats);

#endif