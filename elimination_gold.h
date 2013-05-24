#ifndef _ELIMINATION_GOLD_H_
#define _ELIMINATION_GOLD_H_

float elimination_gold(float *a, float *b, int size, int kernel);
float cpu_kernel_1(float *a, float *b, int size);
float cpu_kernel_2(float *a, float *b, int size);
float cpu_kernel_3(float *a, float *b, int size);

#endif //_ELIMINATION_GOLD_H_
