#ifndef _ELIMINATION_KERNEL_H_
#define _ELIMINATION_KERNEL_H_

void elimination_kernel(float *a, float *b, int n, int kernel);
__global__ void elimination0(float *a, float *b, int n);

#endif //_ELIMINATION_KERNEL_H_
