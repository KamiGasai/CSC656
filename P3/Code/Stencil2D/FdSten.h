#ifndef _FDIFFSTEN_H_
#define _FDIFFSTEN_H_

#define dataAt(DATA, I, J, W) DATA[(I) * (W) + J]

void updateGrid(float *, float *, int, int *, float *, int, int);
void printGrid(float *, int);
void printMid(float g[], int w, int r);
void initGrid(float [], float [], int);
void dumpGrid(float g[], int w);

#endif
