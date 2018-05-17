#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

#define TILE_WIDTH 16 // Tile width
#define WIDTH 2048
__global__ void
matrixMulKernel( float* Md, float* Nd, float* Pd, int Width);

#endif // _MATRIXMUL_H_

