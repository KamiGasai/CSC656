#include <stdio.h>
#include "FdSten.h"

/****

     tu[] is input grid
     u[] is output grid
     w is width of grid

     compute neighbor updates according to stencil

****/

void updateGrid(float u[], float tu[], int w, int off[], float c[],
		int s, int bw) {
  int i, j, k;

  int currIndex;
  float sum;
  for (i=bw; i<w-bw; i++) {
    for (j=bw; j<w-bw; j++) {
      currIndex = i * w + j;
      sum = 0;
      for (k=0; k<s; k++) {
	sum += tu[currIndex + off[k]] * c[k];
      }
      u[currIndex] = sum;
    }
  }
}

void printGrid(float g[], int w) {
  int i, j;

  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      printf("%7.3f ", g[i*w+ j]);
    }
    printf("\n");
  }
}

void dumpGrid(float g[], int w) {
  int i, j;
  FILE *fp;

  fp = fopen("./dump.out", "w");
  
  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      fprintf(fp, "%f ", g[i*w+j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void initGrid(float u0[], float u1[], int w) {
  int i, j;

  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      u0[i*w+j] = 0.;
      u1[i*w+j] = 0.;
    }
  }
}

