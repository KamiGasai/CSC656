/*************************

   File: FdSten.c
   Compile: gcc FdSten.c FdStenUtils.c -O3 -o FdSten -lm
   Use: ./FdSten [input file]

   Performs neighbor updates on 2-D grid according to stencil
   Input file format:

   # cycles
   width of grid (including boundary)
   # points in stencil
   3 integers per stencil data point: i and j offsets, coefficient

   # initial data points
   
   3 integers per data point: i and j indices, data


*************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "FdSten.h"

int main(int arg, char **argv) {
  int width;
  int numCycles;
  int i, j, n;
  float *u0, *u1, *tptr;
  float inTemp;
  int cycle = 0;
  int numInit;

  int numStencil; /* number of points in stencil */
  int *offset;
  float *coeff;
  int borderWidth;
  
  FILE *fp;
  clock_t t1, t2;

  fp = fopen(argv[1], "r");

  fscanf(fp, "%d", &numCycles);
  fscanf(fp, "%d", &width);
  fscanf(fp, "%d", &numStencil);

  offset = (int *) calloc(numStencil, sizeof(int));
  coeff = (float *) calloc(numStencil, sizeof(float));

  borderWidth = 0;
  for (n=0; n<numStencil; n++) {
    fscanf(fp, "%d%d%f", &i, &j, &inTemp);
    offset[n] = i * width + j;
    coeff[n] = inTemp;
    if (abs(i) > borderWidth)
      borderWidth = abs(i);
    if (abs(j) > borderWidth)
      borderWidth = abs(j);
  }
  
  fscanf(fp, "%d", &numInit);
  printf("# cycles %d width %d stencil size %d # initializations %d\n", numCycles, width, numStencil, numInit);

  printf("Border width: %d\n", borderWidth);
  printf("Stencil offsets:\n");
  for (n=0; n<numStencil; n++) {
    printf("%d ", offset[n]);
  }
  printf("\n");
  printf("Stencil coefficients:\n");
  for (n=0; n<numStencil; n++) {
    printf("%f ", coeff[n]);
  }
  printf("\n");

  u0 = calloc(width * width, sizeof(float));
  u1 = calloc(width * width, sizeof(float));

  initGrid(u0, u1, width);

  for (n=0; n<numInit; n++) {
    fscanf(fp, "%d%d%f", &i, &j, &inTemp);
    u1[i * width + j] = inTemp;
  }
  
  //printGrid(u1, width);

  
  t1 = clock();

  for (cycle=0; cycle<numCycles; cycle++) {
    updateGrid(u0, u1, width, offset, coeff, numStencil, borderWidth);
    //printGrid(u0, width);
    tptr = u0;
    u0 = u1;
    u1 = tptr;
  }

  t2 = clock();
  printf("Elapsed CPU time = %.4f\n", (t2-t1)/(double) CLOCKS_PER_SEC);

  dumpGrid(u1, width);

}

