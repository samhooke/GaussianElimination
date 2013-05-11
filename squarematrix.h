#ifndef _SQUAREMATRIX_H_
#define _SQUAREMATRIX_H_

#include <stdio.h>
#include <stdlib.h>

typedef struct {
	int size;
	float* elements;
} SquareMatrix;

SquareMatrix AllocateSquareMatrix(int size, int init);
void DisplaySquareMatrix(SquareMatrix M, int precision);

#endif //_SQUAREMATRIX_H_
