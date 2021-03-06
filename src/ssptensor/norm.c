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

#include <HiParTI.h>
#include "ssptensor.h"
#include <math.h>

double pti_SemiSparseTensorNorm(const ptiSemiSparseTensor *X) {
    double sqnorm = 0;
    ptiNnzIndex i;
    for(i = 0; i < X->nnz; ++i) {
        ptiIndex j;
        for(j = 0; j < X->values.ncols; ++j) {
            double cell_value = X->values.values[i * X->stride + j];
            sqnorm += cell_value * cell_value;
        }
    }
    return sqrt(sqnorm);
}
