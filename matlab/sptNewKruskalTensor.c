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
#include <stdlib.h>
#include "matrix.h"
#include "mex.h"
#include "sptmx.h"

spt_DefineCastArray(spt_mxArrayToSize, size_t)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptNewKruskalTensor", 1, "One", 3, "Three");

    size_t nmodes = mxGetScalar(prhs[0]);
    if(mxGetNumberOfElements(prhs[1]) != nmodes) {
        mexErrMsgIdAndTxt("ParTI:sptNewKruskalTensor", "length of ndims should be nmodes");
    }
    size_t *ndims = spt_mxArrayToSize(prhs[1]);
    size_t rank = mxGetScalar(prhs[2]);

    sptKruskalTensor *ktsr = malloc(sizeof *ktsr);
    int result = sptNewKruskalTensor(ktsr, nmodes, ndims, rank);
    if(result) {
        free(ktsr);
        ktsr = NULL;
    }

    mexCallMATLAB(nlhs, plhs, 0, NULL, "sptKruskalTensor");
    spt_mxSetPointer(plhs[0], 0, ktsr);
}
