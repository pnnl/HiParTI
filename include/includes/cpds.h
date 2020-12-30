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

#ifndef HIPARTI_CPDS_H
#define HIPARTI_CPDS_H


/**
 * CP-ALS
 */
int ptiCpdAls(
  ptiSparseTensor const * const ptien,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  ptiKruskalTensor * ktensor);
int ptiOmpCpdAls(
  ptiSparseTensor const * const ptien,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  const int tk,
  const int use_reduce,
  ptiKruskalTensor * ktensor);
int ptiCudaCpdAls(
  ptiSparseTensor const * const ptien,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  ptiKruskalTensor * ktensor);
int ptiCpdAlsHiCOO(
  ptiSparseTensorHiCOO const * const hitsr,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  ptiRankKruskalTensor * ktensor);
int ptiOmpCpdAlsHiCOO(
  ptiSparseTensorHiCOO const * const hitsr,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  const int tk,
  const int tb,
  int balanced,
  ptiRankKruskalTensor * ktensor);

#endif