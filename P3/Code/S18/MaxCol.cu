// File: MaxCol.cu
// Compile: nvcc MaxCol.cu -o mc
// Run: ./mc [width of matrix] [threads per block]

// Description: finds the max of each column of a randomly generated matrix
// 		in kernel findMax(), each thread finds the max of one column

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define THREADSPERBLOCK 4

int checkArray(int [], int [], int);

__global__ void findMax(int *m, int *rs, int n);

int main(int argc, char **argv)
{
    /* variables for timing */
    cudaEvent_t start, stop;
    float time;

    if (argc != 3) {
       printf("Usage: ./SR [width of matrix] [threads per block]\n");
       exit(0);
    }

    int n = atoi(argv[1]);  // number of matrix rows/cols
    int *hm, // host matrix
        *dm, // device matrix
        *hcs, // host column sums
        *dcs; // device column sums
    int *checkCs;
    int msize = n * n * sizeof(int);  // size of matrix in bytes
    int rssize = n * sizeof(int);
    int threadsPerBlock = atoi(argv[2]); // get threads per block

    if (n % threadsPerBlock != 0) {
       printf("Warning: width of matrix not divisible by # threads per block\n");
    }

    // allocate space for host matrix
    hm = (int *) malloc(msize);  

    // create timer events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // as a test, fill matrix with random integers

    int i, j;
    for (i = 0; i < n; i++) {
       for (j = 0; j < n; j++) {
          hm[i*n+j] = random() % RAND_MAX;
       }
    }

    // compute max of columns on CPU for checking
    checkCs = (int *) malloc(rssize);
    for (i=0; i<n; i++) {
       checkCs[i] = hm[i];
       for (j=0; j<n; j++) {
          if (checkCs[i] < hm[i + j*n])
             checkCs[i] = hm[i + j*n];
       }
    }

    // allocate space for device matrix 
    cudaMalloc((void **)&dm,msize);
    // copy host matrix to device matrix
    cudaMemcpy(dm,hm,msize,cudaMemcpyHostToDevice);
    // allocate host, device rowsum arrays
    hcs = (int *) malloc(rssize);  
    cudaMalloc((void **)&dcs,rssize);

    // record start timestamp
    cudaEventRecord(start, 0);

    // invoke the kernel
    findMax<<<n/threadsPerBlock,threadsPerBlock>>>(dm,dcs,n);
    // wait for kernel to finish
    cudaThreadSynchronize();
    // copy row vector from device to host
    cudaMemcpy(hcs,dcs,rssize,cudaMemcpyDeviceToHost);

    // get elapsed time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Elapsed time = %f\n", time);

    // check results
    int diff = checkArray(hcs, checkCs, n);
    if (diff == 0) {
       printf("Arrays match\n");
    }
    else {
       printf("Arrays do not match\n");
    }


    // clean up
    free(hm);
    cudaFree(dm);
    free(hcs);
    cudaFree(dcs);
}

int checkArray(int x[], int y[], int size) {
   int i;
   int numDiff = 0;
   
   for (i=0; i<size; i++) {
      if (x[i] != y[i]) {
         numDiff++;
      }
   }
   return numDiff;
}

// findMax(int *m, int *cs, int n)
// m: n x n matrix (input)
// cs: cs[i] contains max of columnn i of m (output)
// n: number of elements in each row/column of m

__global__ void findMax(int *m, int *cs, int n)
{
   // your code goes here
}
