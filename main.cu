#include <stdio.h>
#include <stdlib.h>
#include "squarematrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"

#define MATRIX_SIZE 3

int main() {
	SquareMatrix input = AllocateSquareMatrix(MATRIX_SIZE, 1);
	SquareMatrix output = AllocateSquareMatrix(MATRIX_SIZE, 0);

	printf("Input:\n");
	DisplaySquareMatrix(input, 4);

	elimination_gold(input, output);

	printf("Output 2:\n");
	DisplaySquareMatrix(output, 4);

	return 0;
}
