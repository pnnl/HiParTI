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

#ifndef PARTI_SSPTENSOR_H
#define PARTI_SSPTENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <HiParTI.h>
 

int pti_SemiSparseTensorAppend(ptiSemiSparseTensor *tsr, const ptiIndex indices[], ptiValue value);
int pti_SemiSparseTensorCompareIndices(const ptiSemiSparseTensor *tsr1, ptiNnzIndex ind1, const ptiSemiSparseTensor *tsr2, ptiNnzIndex ind2);
int pti_SemiSparseTensorMergeValues(ptiSemiSparseTensor *tsr);

double pti_SemiSparseTensorNorm(const ptiSemiSparseTensor *X);

int pti_SemiSparseTensorSetMode(
    ptiSemiSparseTensor       *dest,
    const ptiSemiSparseTensor *src,
    ptiIndex                    newmode
);
int pti_SemiSparseTensorToSparseMatrixCSR(
    ptiValue                  *csrVal,
    ptiNnzIndex                        *csrRowPtr,
    ptiIndex                        *csrColInd,
    const ptiSemiSparseTensor  *tsr
);

#ifdef __cplusplus
}
#endif

#endif
