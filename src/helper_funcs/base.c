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
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>


char * ptiBytesString(uint64_t const bytes)
{
  double size = (double)bytes;
  int suff = 0;
  const char *suffix[5] = {"B", "KiB", "MiB", "GiB", "TiB"};
  while(size > 1024 && suff < 4) {
    size /= 1024.;
    ++suff;
  }
  char * ret = NULL;
  if(asprintf(&ret, "%0.2f %s", size, suffix[suff]) == -1) {
    fprintf(stderr, "SPT: asprintf failed with %" PRIu64 " bytes.\n", bytes);
    ret = NULL;
  }
  return ret;
}


ptiValue ptiRandomValue(void)
{
  ptiValue v =  3.0 * ((ptiValue) rand() / (ptiValue) RAND_MAX);
  if(rand() % 2 == 0) {
    v *= -1;
  }
  return v;
}
