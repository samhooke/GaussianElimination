#include <stdio.h>
#include <stdlib.h>
#include "squarematrix.h"

#define MATRIX_SIZE 3

int main() {
	SquareMatrix input = AllocateSquareMatrix(MATRIX_SIZE, 1);
	SquareMatrix output = AllocateSquareMatrix(MATRIX_SIZE, 0);

	DisplaySquareMatrix(input, 4);

	return 0;
}
