#ifndef _ELIMINATION_KERNEL_H_
#define _ELIMINATION_KERNEL_H_

#include "check.h"

float elimination_kernel(float *a, float *b, int n, int kernel);
__global__ void elimination0(float *a, float *b, int n);
__global__ void elimination1(float *a, float *b, int n);
__global__ void elimination2(float *a, float *b, int n);
__global__ void elimination3(float *a, float *b, int n);
__global__ void elimination4(float *a, float *b, int n);
__global__ void elimination5(float *a, float *b, int n);

#endif //_ELIMINATION_KERNEL_H_
