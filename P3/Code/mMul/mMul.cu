
#include <stdio.h>
#include "matrixMul.h"

///////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: P = M * N
///////////////////////////////////////////////////////////////////////////////

__global__ void
matrixMulKernel( float* Md, float* Nd, float* Pd, int Width)
{

    // Thread index
    float Psub = 0.;
    int Row = threadIdx.y;
    int Col = threadIdx.x;

    for (int k = 0; k < Width; ++k) {
        Psub += Md[Row * Width + k] * Nd[k * Width + Col];
    }

    Pd[Row * Width + Col] = Psub;
}

