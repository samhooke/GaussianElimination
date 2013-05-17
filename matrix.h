#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdio.h>
#include <stdlib.h>

float* matrix_generate(int size, int type);
float matrix_compare_b(float *m, float *n, int size, float tolerance);

#endif //_MATRIX_H_
