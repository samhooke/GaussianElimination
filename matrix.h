#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdio.h>
#include <stdlib.h>

typedef struct {
	float* a;
	float* b;
	int size;
} Matrix;

Matrix matrix_generate(int size, int type);

#endif //_MATRIX_H_
