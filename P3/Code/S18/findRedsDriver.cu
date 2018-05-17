/****                                                                           
     File: findRedsDriver.cu
     Date: 5/3/2018
     By: Bill Hsu
     Compile: nvcc findRedsDriver.cu -o frgpu
     Run: ./frgpu
                                                     
****/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>

#define NUMPARTICLES 32768
#define NEIGHBORHOOD .05
#define THREADSPERBLOCK 128

void initPos(float *);
float findDistance(float *, int, int);
__device__ float findDistanceGPU(float *, int, int);
void dumpResults(int index[]);

__global__ void findRedsGPU(float *p, int *numI);

int main() {
  cudaEvent_t start, stop;
  float time;
  
  float *pos;
  int *numReds;

  pos = (float *) malloc(NUMPARTICLES * 4 * sizeof(float));
  numReds = (int *) malloc(NUMPARTICLES * sizeof(int));

  initPos(pos);

  // your code to allocate device arrays for pos and numReds go here




  // create timer events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  /* invoke kernel findRedsGPU here */

  cudaThreadSynchronize();

  // your code to copy results to numReds[] go here 



  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Elapsed time = %f\n", time);

  dumpResults(numReds);

}

void initPos(float *p) {

  // your code for initializing pos goes here

}

__device__ float findDistanceGPU(float *p, int i, int j) {

  // your code for calculating distance for particle i and j

}

__global__ void findRedsGPU(float *p, int *numI) {

  // your code for counting red particles goes here

}


void dumpResults(int index[]) {
  int i;
  FILE *fp;

  fp = fopen("./dump.out", "w");
  
  for (i=0; i<NUMPARTICLES; i++) {
    fprintf(fp, "%d %d\n", i, index[i]);
  }

  fclose(fp);
}
