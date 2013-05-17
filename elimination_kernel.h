#ifndef _ELIMINATION_KERNEL_H_
#define _ELIMINATION_KERNEL_H_

#include "check.h"

// Only required for debugging
#include "elimination_gold.h"

float elimination_kernel(float *a, float *b, int size, int kernel);
__global__ void elimination0(float *a, float *b, int size);
__global__ void elimination1(float *a, float *b, int size);
__global__ void elimination2(float *a, float *b, int size);
__global__ void elimination3(float *a, float *b, int size);
__global__ void elimination4(float *a, float *b, int size);
__global__ void elimination5(float *a, float *b, int size);
__global__ void elimination6(float *a, float *b, int size, int pivot);
__global__ void elimination7(float *a, float *b, int size, int pivot);
__global__ void elimination8_1(float *a, int size, int pivot);
__global__ void elimination8_2(float *a, int size);
__global__ void elimination9(float *a, int size);
__global__ void elimination10(float *a, int size);
__global__ void elimination11_1(float *a, int size, int pivot);
__global__ void elimination11_2(float *a, int size);
__global__ void elimination12_1(float *a, int size, int pivot);
__global__ void elimination12_2(float *a, int size, int pivot);
__global__ void elimination13_1(float *a, int size, int pivot);
__global__ void elimination13_2(float *a, int size, int pivot);
__global__ void elimination14_1(float *a, int size, int pivot);
__global__ void elimination14_2(float *a, int size, int pivot);


#endif //_ELIMINATION_KERNEL_H_
