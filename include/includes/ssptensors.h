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

#ifndef HIPARTI_SSPTENSORS_H
#define HIPARTI_SSPTENSORS_H


/**
 * epsilon is a small positive value, every -epsilon < x < x would be considered as zero
 */
int ptiSemiSparseTensorToSparseTensor(ptiSparseTensor *dest, const ptiSemiSparseTensor *src, ptiValue epsilon);

int ptiNewSemiSparseTensor(ptiSemiSparseTensor *tsr, ptiIndex nmodes, ptiIndex mode, const ptiIndex ndims[]);
int ptiCopySemiSparseTensor(ptiSemiSparseTensor *dest, const ptiSemiSparseTensor *src);
void ptiFreeSemiSparseTensor(ptiSemiSparseTensor *tsr);
int ptiSparseTensorToSemiSparseTensor(ptiSemiSparseTensor *dest, const ptiSparseTensor *src, ptiIndex mode);
int ptiSemiSparseTensorSortIndex(ptiSemiSparseTensor *tsr);

int ptiNewSemiSparseTensorGeneral(ptiSemiSparseTensorGeneral *tsr, ptiIndex nmodes, const ptiIndex ndims[], ptiIndex ndmodes, const ptiIndex dmodes[]);
void ptiFreeSemiSparseTensorGeneral(ptiSemiSparseTensorGeneral *tsr);

/**
 * Set indices of a semi-sparse according to a reference sparse
 * Call ptiSparseTensorSortIndexAtMode on ref first
 */
int ptiSemiSparseTensorSetIndices(ptiSemiSparseTensor *dest, ptiNnzIndexVector *fiberidx, ptiSparseTensor *ref);


/**
 * Semi-sparse tensor times a dense matrix (TTM)
 * Input: semi-sparse tensor X[I][J][K], dense matrix U[I][R}, mode n={0, 1, 2}
 * Output: sparse tensor Y[I][J][R] (e.g. n=2)
 */
int ptiSemiSparseTensorMulMatrix(ptiSemiSparseTensor *Y, const ptiSemiSparseTensor *X, const ptiMatrix *U, ptiIndex mode);
int ptiCudaSemiSparseTensorMulMatrix(ptiSemiSparseTensor *Y, const ptiSemiSparseTensor *X, const ptiMatrix *U, ptiIndex mode);
#endif