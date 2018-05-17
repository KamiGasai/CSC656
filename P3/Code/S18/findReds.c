/****

     File: findReds.c
     Date: 5/3/2018
     By: Bill Hsu
     Compile: gcc findReds.c -O3 -o findReds -lm
     Run: ./findReds

****/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUMPARTICLES 32768
#define NEIGHBORHOOD .05

void initPos(float *);
float findDistance(float *, int, int);
void findRedsCPU(float *p, int *numI);
void dumpResults(int index[]);

int main() {
  int i;
  float *pos;
  int *numReds;

  /* set up timer */
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double t0 = tv.tv_sec*1e6 + tv.tv_usec;

  pos = (float *) malloc(NUMPARTICLES * 4 * sizeof(float));
  numReds = (int *) malloc(NUMPARTICLES * sizeof(int));

  initPos(pos);

  findRedsCPU(pos, numReds);

  gettimeofday(&tv, NULL);
  double t1 = tv.tv_sec*1e6 + tv.tv_usec;
  printf("Elapsed CPU time = %f ms\n", (t1-t0)*1e-3);

  dumpResults(numReds);

}

void initPos(float *p) {
  int i;
  int roll;
  for (i=0; i<NUMPARTICLES; i++) {
    p[i*4] = rand() / (float) RAND_MAX;
    p[i*4+1] = rand() / (float) RAND_MAX;
    p[i*4+2] = rand() / (float) RAND_MAX;
    roll = rand() % 3;
    if (roll == 0)
      p[i*4+3] = 0xff0000;
    else if (roll == 1)
      p[i*4+3] = 0x00ff00;
    else
      p[i*4+3] = 0x0000ff;
  }
}

float findDistance(float *p, int i, int j) {
  float dx, dy, dz;

  dx = p[i*4] - p[j*4];
  dy = p[i*4+1] - p[j*4+1];
  dz = p[i*4+2] - p[j*4+2];

  return(sqrt(dx*dx + dy*dy + dz*dz));
}

void findRedsCPU(float *p, int *numI) {
  int i, j;
  float distance;

  for (i=0; i<NUMPARTICLES; i++) {
    numI[i] = 0;
    for (j=0; j<NUMPARTICLES; j++) {
      if (i!=j) {
	/* calculate distance between particles i, j */
	distance = findDistance(p, i, j);
	/* if distance < r and color is red, increment count */
	if (distance < NEIGHBORHOOD && p[j*4+3] == 0xff0000) {
	  numI[i]++;
	}
      }
    }
  }

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
