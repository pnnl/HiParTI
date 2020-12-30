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

// This file will be compiled only if CUDA is not enabled,
// or cuda_timer.cu will be compiled instead.

#include <HiParTI.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

struct ptiTagTimer {
    int use_cuda;
    struct timespec start_timespec;
    struct timespec stop_timespec;
};

int ptiNewTimer(ptiTimer *timer, int use_cuda) {
    *timer = (ptiTimer) malloc(sizeof **timer);
    (*timer)->use_cuda = use_cuda;
    if(use_cuda) {
        pti_CheckError(3 + PTIERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    }
    return 0;
}

int ptiStartTimer(ptiTimer timer) {
    int result;
    if(timer->use_cuda) {
        pti_CheckError(3 + PTIERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    } else {
        result = clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
        pti_CheckOSError(result, "Timer New");
    }
    return 0;
}

int ptiStopTimer(ptiTimer timer) {
    int result;
    if(timer->use_cuda) {
        pti_CheckError(3 + PTIERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    } else {
        result = clock_gettime(CLOCK_MONOTONIC, &timer->stop_timespec);
        pti_CheckOSError(result, "Timer New");
    }
    return 0;
}

double ptiElapsedTime(const ptiTimer timer) {
    if(timer->use_cuda) {
        return NAN;
    } else {
        return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
            + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    }
}

double ptiPrintElapsedTime(const ptiTimer timer, const char *name) {
    double elapsed_time = ptiElapsedTime(timer);
    fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    return elapsed_time;
}


double ptiPrintAverageElapsedTime(const ptiTimer timer, const int niters, const char *name) {
    double elapsed_time = ptiElapsedTime(timer) / niters;
    fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    return elapsed_time;
}


int ptiFreeTimer(ptiTimer timer) {
    if(timer->use_cuda) {
        pti_CheckError(3 + PTIERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    }
    free(timer);
    return 0;
}

double ptiPrintGFLOPS(const double elapsed_time, const ptiNnzIndex flops, const char *name) {
    double gflops = flops / elapsed_time / 1e9;
    fprintf(stdout, "[%s]: %.3lf GFLOPS\n", name, gflops);
    return gflops;
}
