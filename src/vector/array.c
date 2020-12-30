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
#include <string.h>


ptiNnzIndex ptiMaxNnzIndexArray(
  ptiNnzIndex const * const indices,
  ptiNnzIndex const size)
{
  ptiNnzIndex max = indices[0];
  for(ptiNnzIndex i=1; i < size; ++i) {
    if(indices[i] > max) {
      max = indices[i];
    }
  }
  return max;
}


ptiIndex ptiMaxIndexArray(
  ptiIndex const * const indices,
  ptiNnzIndex const size)
{
  ptiIndex max = indices[0];
  for(ptiNnzIndex i=1; i < size; ++i) {
    if(indices[i] > max) {
      max = indices[i];
    }
  }
  return max;
}


static inline int pti_PairCompareIndices(ptiKeyValuePair const * kvarray, ptiIndex loc1, ptiIndex loc2) {

    if(kvarray[loc1].value < kvarray[loc2].value) {
        return -1;
    } else if(kvarray[loc1].value > kvarray[loc2].value) {
        return 1;
    } else {
        return 0;
    }
}


static inline void pti_SwapPairs(ptiKeyValuePair * kvarray, ptiIndex const ind1, ptiIndex const ind2) {
    
    ptiIndex eleind1 = kvarray[ind1].key;
    kvarray[ind1].key = kvarray[ind2].key;
    kvarray[ind2].key = eleind1;

    ptiIndex val1 = kvarray[ind1].value;
    kvarray[ind1].value = kvarray[ind2].value;
    kvarray[ind2].value = val1;
}

static void pti_QuickSortPairArray(ptiKeyValuePair * kvarray, ptiIndex const l, ptiIndex const r)
{
    ptiIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(pti_PairCompareIndices(kvarray, i, p) < 0) {
            ++i;
        }
        while(pti_PairCompareIndices(kvarray, p, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        pti_SwapPairs(kvarray, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }

    pti_QuickSortPairArray(kvarray, l, i);
    pti_QuickSortPairArray(kvarray, i, r);

}

/**
 * Increasingly sort an key-value pair array in type ptiIndex.
 *
 * @param array a pointer to an array to be sorted,
 * @param length number of values 
 *
 */
void ptiPairArraySort(ptiKeyValuePair * kvarray, ptiIndex const length)
{
    pti_QuickSortPairArray(kvarray, 0, length);
}

/// The return value has a small chance to overflow.
long int ptiInArray(ptiIndex * array, ptiNnzIndex len, ptiIndex value)
{
  int result = -1;
  for (ptiNnzIndex i = 0; i < len; ++i)
    if(value == array[i])
      return i;
  return result;
}