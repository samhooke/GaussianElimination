#include "squarematrix.h"
#include <stdio.h>
#include <stdlib.h>

// Allocates space for a SquareMatrix, and populates with values
// size: width/height of the square matrix
// init: 0 = populate matrix with 0s
//       1 = populate with random numbers
SquareMatrix AllocateSquareMatrix(int size, int init) {
	SquareMatrix M;
	M.size = size;
	M.elements = (float*) malloc(size * sizeof(float));

	for (unsigned int i = 0; i < M.size * M.size; i++)
		M.elements[i] = (init == 0) ? 0.0f : 3 * (rand() / (float) RAND_MAX);

	return M;
}

// Prints out the elements of the supplied SquareMatrix
// M: The SquareMatrix to print
// precision: Precision with which to print the floats
void DisplaySquareMatrix(SquareMatrix M, int precision) {
	char fmt[10];
	sprintf(fmt, "%%.%df", precision);

	for (unsigned int y = 0; y < M.size; y++) {
		for (unsigned int x = 0; x < M.size; x++) {
			printf(fmt, M.elements[y * M.size + x]);
			if (x < M.size - 1) {
				printf(" ");
			}
		}
		printf("\n");
	}
}
