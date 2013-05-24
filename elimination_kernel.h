#ifndef _ELIMINATION_KERNEL_H_
#define _ELIMINATION_KERNEL_H_

#include "check.h"

// Only required for debugging
#include "elimination_gold.h"

float elimination_kernel(float *a, float *b, int size, int kernel);
__global__ void gpu_kernel_1(float *a, float *b, int size);
__global__ void gpu_kernel_2(float *a, float *b, int size);
__global__ void gpu_kernel_3(float *a, float *b, int size);
__global__ void gpu_kernel_4(float *a, float *b, int size);
__global__ void gpu_kernel_5a(float *a, int size, int pivot);
__global__ void gpu_kernel_5b(float *a, int size);
__global__ void gpu_kernel_6a(float *a, int size, int pivot);
__global__ void gpu_kernel_6b(float *a, int size);
__global__ void gpu_kernel_7a(float *a, int size, int pivot);
__global__ void gpu_kernel_7b(float *a, int size, int pivot);
__global__ void gpu_kernel_8a(float *a, int size, int pivot);
__global__ void gpu_kernel_8b(float *a, int size, int pivot);
__global__ void gpu_kernel_9a(float *a, int size, int pivot);
__global__ void gpu_kernel_9b(float *a, int size, int pivot);
__global__ void gpu_kernel_10a(float *a, int size, int pivot);
__global__ void gpu_kernel_10b(float *a, int size, int pivot);
__global__ void gpu_kernel_11a(float *a, int size, int pivot);
__global__ void gpu_kernel_11b(float *a, int size, int pivot);
__global__ void gpu_kernel_12a(float *a, float *b, int size, int pivot);
__global__ void gpu_kernel_12b(float *a, float *b, int size, int pivot);
__global__ void gpu_kernel_13a(float *a, float *b, int size, int pivot);
__global__ void gpu_kernel_13b(float *a, float *b, int size, int pivot);
__global__ void gpu_kernel_14a(float *a, float *b, int size, int pivot);
__global__ void gpu_kernel_14b(float *a, float *b, int size, int pivot);
__global__ void gpu_kernel_15a(float *a, float *b, int size, int pivot);
__global__ void gpu_kernel_15b(float *a, float *b, int size, int pivot);
__global__ void gpu_kernel_16a(float *a, float *b, int size, int pivot, int xoffset);
__global__ void gpu_kernel_16b(float *a, float *b, int size, int pivot, int xoffset);
__global__ void gpu_kernel_17a(float *a, float *b, int size, int pivot, int xoffset);
__global__ void gpu_kernel_17b(float *a, float *b, int size, int pivot, int xoffset);

#endif //_ELIMINATION_KERNEL_H_
