//
//	matrixMul.cu
//
//	Compile: nvcc matrixMul.cu matrixMul_gold.cpp -O3 -lm -o mMul
//	Usage: ./mMul

// includes, kernels
// #include "mMul.cu"
#include <stdio.h>
#include "matrixMul.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);
__global__ void
matrixMulKernel( float* Md, float* Nd, float* Pd, int Width);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int);

texture <float> texM;
texture <float> texN;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices M and N
    unsigned int size_M = WIDTH * WIDTH;
    unsigned int mem_size_M = sizeof(float) * size_M;
    float* h_M = (float*)malloc(mem_size_M);
    unsigned int size_N = WIDTH * WIDTH;
    unsigned int mem_size_N = sizeof(float) * size_N;
    float* h_N = (float*)malloc(mem_size_N);

    // initialize host memory
    randomInit(h_M, size_M);
    randomInit(h_N, size_N);

    // allocate device memory
    float* d_M;
    cudaMalloc((void**) &d_M, mem_size_M);
    float* d_N;
    cudaMalloc((void**) &d_N, mem_size_N);

    cudaBindTexture(NULL, texM, d_M, WIDTH*WIDTH*sizeof(float));
    cudaBindTexture(NULL, texN, d_N, WIDTH*WIDTH*sizeof(float));

    cudaEventRecord(start, 0);

    // copy host memory to device
    cudaMemcpy(d_M, h_M, mem_size_M,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, mem_size_N,
                              cudaMemcpyHostToDevice);

    // allocate device memory for result
    unsigned int size_P = WIDTH * WIDTH;
    unsigned int mem_size_P = sizeof(float) * size_P;
    float* d_P;
    cudaMalloc((void**) &d_P, mem_size_P);

    // allocate host memory for the result
    float* h_P = (float*) malloc(mem_size_P);
    
    // setup execution parameters
    dim3 block(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH);
    dim3 threads(TILE_WIDTH, TILE_WIDTH);

    matrixMulKernel<<< block, threads >>>(d_M, d_N, d_P, WIDTH);
    cudaThreadSynchronize();
    
    // copy result from device to host
    cudaMemcpy(h_P, d_P, mem_size_P,
                              cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Elapsed time = %f ms\n", time);

    // compute reference solution
    float* reference = (float*)malloc(mem_size_P);
    computeGold(reference, h_M, h_N, WIDTH);

    // check result
    printDiff(reference, h_P, WIDTH, WIDTH, 100, 1.0e-5f);

    // clean up memory
    cudaUnbindTexture(texM);
    cudaUnbindTexture(texN);
    free(h_M);
    free(h_N);
    free(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    cudaThreadExit();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int j,k;
    int error_count=0;
    for (j = 0; j < height; j++) 
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }
        for (int i = 0; i < width; i++) 
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]) / data1[k];
            if (fDiff > fListTol) 
            {                
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    printf(" \n  Total Errors = %d\n\n", error_count);
}

__global__ void
matrixMulKernel( float* Md, float* Nd, float* Pd, int Width)
{

    // Thread index
    float Psub = 0.;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;

    for (int k = 0; k < Width; ++k) {
        float m = tex1Dfetch(texM, k + Row * Width);
	float n = tex1Dfetch(texN, Col + k * Width);
        //Psub += Md[Row * Width + k] * Nd[k * Width + Col];
	Psub += m * n;
    }
    Pd[Row * Width + Col] = Psub;

}
