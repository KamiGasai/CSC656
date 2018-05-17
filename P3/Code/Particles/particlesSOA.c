/*******************************************

    File: particlesSOA.c
    Compile: gcc particlesSOA.c -o particlesSOA -O3 -lm
    Run: ./particlesSOA [input file]

    Simple particle system with collision detection

*******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define NUMCYCLES 100 /* number of cycles to run simulation */

/* macros for accessing array of 2-D positions, velocities and accelerations */
/* DATA is particle array */
/* POINTINDEX is particle id */
/* NP is number of particles */
/* x is DIMINDEX=0, y is DIMINDEX=1 */

#define pos(DATA, POINTINDEX, DIMINDEX, NP) DATA[(DIMINDEX)*(NP) + POINTINDEX]
#define vel(DATA, POINTINDEX, DIMINDEX, NP) DATA[(DIMINDEX+2)*(NP) + POINTINDEX]
#define acc(DATA, POINTINDEX, DIMINDEX, NP) DATA[(DIMINDEX+4)*(NP) + POINTINDEX]

#define RADIUS 8    /* radius of ball */
#define SPACING 24  /* spacing between balls */
#define WIDTH 800   /* width of display grid */
#define HEIGHT 600  /* height of display grid */
#define DAMP .99    /* velocity damp factor */

struct line {
  int particleIndex;
  float velX;
  float velY;
};

void initGrid(float *, int, int, int, int);
void initVels(float *, int, struct line *, int);
void dumpGrid(float *, int);
void updateParticles(float *, int);
void collisionDetect(float *, int np);
void cd1(float *p1, float *p, int i, int np, int myPosX, int myPosY);

int main(int argc, char **argv)
{
  int i;
  struct line *l;

  if (argc != 2) {
    printf("Usage: ./particles [script]\n");
    return 0;
  }

  /* read input file of initial velocities */
  FILE *fp = fopen(argv[1], "r");  
  int pIndex;
  float vX, vY;
  int numScriptLine;
  int temp;

  fscanf(fp, "%d", &numScriptLine);
  l = (struct line *) calloc(numScriptLine, sizeof(struct line));
  for (i=0; i<numScriptLine; i++) {
    temp = fscanf(fp, "%d%f%f", &pIndex, &vX, &vY);
    if (temp != 3) {
      printf("Error! Script file format incorrect\n");
      return(0);
    }
    l[i].particleIndex = pIndex;
    l[i].velX = vX;
    l[i].velY = vY;
    
  }

  for (i=0; i<numScriptLine; i++) {
    printf("%d %f %f\n", l[i].particleIndex, l[i].velX, l[i].velY);
  }

  int n = (WIDTH / SPACING) * (HEIGHT / SPACING);
  printf("%d particles\n", n);

  float *hParticle; // host array of particle position, velocity, acceleration
  int hPSize = 6 * n * sizeof(float);  // size of matrix in bytes
  // allocate space for host array
  hParticle = (float *) malloc(hPSize);  

  // initialize velocities and accelerations

  for (i = 0; i < n; i++) {
    vel(hParticle, i, 0, n) = 0;
    vel(hParticle, i, 1, n) = 0;
    acc(hParticle, i, 0, n) = 0;
    acc(hParticle, i, 1, n) = 0;
  }

  /* place particles in an evenly spaced grid */
  initGrid(hParticle, n, SPACING, WIDTH, HEIGHT);

  /* initialize particle velocities from input file */
  initVels(hParticle, n, l, numScriptLine);

  // set up timer
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double t0 = tv.tv_sec*1e6 + tv.tv_usec;

  /* run simulation for NUMCYCLES steps */
  for (i=0; i<NUMCYCLES; i++) {
    updateParticles(hParticle, n);
    collisionDetect(hParticle, n);
  }

  gettimeofday(&tv, NULL);
  double t1 = tv.tv_sec*1e6 + tv.tv_usec;
  printf("Elapsed CPU time = %f ms\n", (t1-t0)*1e-3);
  
  // dump all particle positions and velocities to file
  dumpGrid(hParticle, n);

  // clean up
  free(hParticle);
}

// hp: pointer to particle array
// nhp: number of particles in array
// for particle i, 
//     read its position, velocity and acceleration
//     update its velocity and position

void updateParticles(float *hp, int nhp) {
  int i;
  float velX, velY, newPosX, newPosY;
  for (i=0; i<nhp; i++) {

    velX = vel(hp, i, 0, nhp);
    velY = vel(hp, i, 1, nhp);
    velX = velX + acc(hp, i, 0, nhp);
    velY = velY + acc(hp, i, 1, nhp);

    newPosX = pos(hp, i, 0, nhp) + velX;

    /* check for bouncing off edges of display grid */

    if (newPosX < RADIUS) {
      newPosX = RADIUS;
      velX = - velX;
    }
    if (newPosX >= WIDTH - RADIUS) {
      newPosX = WIDTH - RADIUS - 1;
      velX = - velX;
    }

    newPosY = pos(hp, i, 1, nhp) + velY;
    if (newPosY < RADIUS) {
      newPosY = RADIUS;
      velY = - velY;
    }
    if (newPosY >= HEIGHT - RADIUS) {
      newPosY = HEIGHT - RADIUS - 1;
      velY = - velY;
    }

    /* end check for edges */

    velX *= DAMP;
    velY *= DAMP;

    vel(hp, i, 0, nhp) = velX;
    vel(hp, i, 1, nhp) = velY;
    
    pos(hp, i, 0, nhp) = newPosX;
    pos(hp, i, 1, nhp) = newPosY;

  }
}

// p: array of particles
// np: number of particles in array
// for each particle i,
//     read its position
//     check with every other particle j for collision
//     update acceleration of particle i
void collisionDetect(float *p, int np) {
  float forceX;
  float forceY;
  float relPosX;
  float relPosY;
  float normX;
  float normY;
  float d;
  int i, j;

  for (i=0; i<np; i++) {  
    forceX = 0;
    forceY = 0;
    for (j= 0; j<np; j++) {
      if (i == j) continue;
      relPosX = pos(p, j, 0, np) - pos(p, i, 0, np);
      relPosY = pos(p, j, 1, np) - pos(p, i, 1, np);
      d = sqrt(relPosX * relPosX + relPosY * relPosY);

      if (d <= 2 * RADIUS) {
        // this is from collideSpheres() in CUDA particles example
        normX = relPosX / d;
        normY = relPosY / d;
        
        forceX -= .2 * (2 * RADIUS - d) * normX;
        forceY -= .2 * (2 * RADIUS - d) * normY;
      }
    }
    acc(p, i, 0, np) = forceX;
    acc(p, i, 1, np) = forceY;
  }
}

/* place particles evenly spaced in display grid */
void initGrid(float *p, int np, int space, int w, int h) {
  int i;
  float currX = space/2;
  float currY = space/2;

  for (i=0; i<np; i++) {
    pos(p, i, 0, np) = currX;
    pos(p, i, 1, np) = currY;

    currX = currX + space;
    if (currX >= w) {
      currX = space/2;
      currY = currY + space;
    }
  }
}

/* dump all particles to file */
void dumpGrid(float *p, int np) {
  int i;
  FILE *fp;

  fp = fopen("dump.out", "w");
  
  for (i=0; i<np; i++) {
    fprintf(fp, "%d %.4f %.4f %.4f %.4f\n", i, pos(p, i, 0, np), pos(p, i, 1, np),
	    vel(p, i, 0, np), vel(p, i, 1, np));
  }
  fclose(fp);
}

void initVels(float *p, int np, struct line *vl, int vln) {
  int i;
  for (i=0; i<vln; i++) {
    vel(p, vl[i].particleIndex, 0, np) = vl[i].velX;
    vel(p, vl[i].particleIndex, 1, np) = vl[i].velY;
  }
}
