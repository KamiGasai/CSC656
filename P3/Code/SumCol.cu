#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

int checkArray(int [], int [], int);

// CUDA example:  finds column sums of an integer matrix m

// find1elt() finds the sum of one column of the nxn matrix m, storing the
// result in the corresponding position in the rowsum array rs; matrix
// stored as 1-dimensional, row-major order

__global__ void find1elt(int *m, int *rs, int n);

int main(int argc, char **argv)
{
    /* variables for timing */
    cudaEvent_t start, stop;
    float time;

    if (argc != 3) {
       printf("Usage: ./SC [width of matrix] [threads per block]\n");
       exit(0);
    }

    int n = atoi(argv[1]);  // number of matrix rows/cols
    int *hm, // host matrix
        *dm, // device matrix
        *hrs, // host rowsums
        *drs; // device rowsums
    int *checkRs;
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

    // as a test, fill matrix with consecutive integers
    int t = 0,i,j;
    for (i = 0; i < n; i++) {
       for (j = 0; j < n; j++) {
          hm[i*n+j] = t++;
       }
    }

    // compute array of sums on CPU for checking
    checkRs = (int *) malloc(rssize);
    for (i=0; i<n; i++) {
       checkRs[i] = 0;
       for (j=0; j<n; j++) {
          checkRs[i] += hm[j*n+i];
       }
    }

    // allocate space for device matrix 
    cudaMalloc((void **)&dm,msize);
    // copy host matrix to device matrix
    cudaMemcpy(dm,hm,msize,cudaMemcpyHostToDevice);
    // allocate host, device rowsum arrays
    hrs = (int *) malloc(rssize);  
    cudaMalloc((void **)&drs,rssize);

    // record start timestamp
    cudaEventRecord(start, 0);

    // invoke the kernel
    find1elt<<<n/threadsPerBlock,threadsPerBlock>>>(dm,drs,n);
    // wait for kernel to finish
    cudaThreadSynchronize();
    // copy row vector from device to host
    cudaMemcpy(hrs,drs,rssize,cudaMemcpyDeviceToHost);

    // get elapsed time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Elapsed time = %f\n", time);

    // check results
    int diff = checkArray(hrs, checkRs, n);
    if (diff == 0) {
       printf("Arrays match\n");
    }
    else {
       printf("Arrays do not match\n");
    }


    // clean up
    free(hm);
    cudaFree(dm);
    free(hrs);
    cudaFree(drs);
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

__global__ void find1elt(int *m, int *rs, int n)
{
   // this thread will handle row # rownum
   int colnum = blockDim.x * blockIdx.x + threadIdx.x;
   int sum = 0;
   for (int    k = 0; k < n; k++)
      sum += m [k*n + colnum];
   rs[colnum] = sum;
}
