#ifndef _ELIMINATION_KERNEL_H_
#define _ELIMINATION_KERNEL_H_

#include "squarematrix.h"

__global__ void elimination(SquareMatrix M, SquareMatrix N);

#endif //_ELIMINATION_KERNEL_H_
