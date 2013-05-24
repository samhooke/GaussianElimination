#ifndef _ELIMINATION_CPU_H_
#define _ELIMINATION_CPU_H_

float elimination_cpu(float *a, float *b, int size, int kernel);
float cpu_kernel_1(float *a, float *b, int size);
float cpu_kernel_2(float *a, float *b, int size);
float cpu_kernel_3(float *a, float *b, int size);

#endif //_ELIMINATION_CPU_H_
