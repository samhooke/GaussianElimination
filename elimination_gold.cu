#include "elimination_gold.h"

// Performs Gauss-Jordan elimination on the CPU; [A]{x]={b}
// Inputs:
//   a -> [A], the matrix of coefficients of size 'n' by 'n'
//   b -> {b}, the vertical matrix of results
//   n -> width/height of 'a', and height of 'b'
// Outputs:
//   Modifies 'a' into the identity matrix
//   Modifies 'b' into the solution for {x}
void elimination_gold(float *a, float *b, int n) {
#define element(_x, _y) (*(a + ((_y) * (n) + (_x))))
	unsigned int xx, yy, rr;
	float c;

	for (yy = 0; yy < n; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1
		for (xx = 0; xx < n; xx++)
			element(xx, yy) /= pivot;
		b[yy] /= pivot;

#ifdef DEBUG
		printf("Matrix (Stage 1; Column %d):\n", yy);
		elimination_gold_print_matrix(a, b, n);
#endif

		// Make all other values in the pivot column be zero
		for (rr = 0; rr < n; rr++) {
			if (rr != yy) {
				c = element(yy, rr);
				for (xx = 0; xx < n; xx++)
					element(xx, rr) -= c * element(xx, yy);
				b[rr] -= c * b[yy];
			}
		}

#ifdef DEBUG
		printf("Matrix (Stage 2; Column %d):\n", yy);
		elimination_gold_print_matrix(a, b, n);
#endif
	}
#undef element
}

// Prints a matrix in the format of [A]{b}
// Inputs:
//   a -> [A], the matrix of coefficients of size 'n' by 'n'
//   b -> {b}, the vertical matrix of results
//   n -> width/height of 'a', and height of 'b'
// Outputs:
//   Prints out the matrix as a nicely formatted table
void elimination_gold_print_matrix(float *a, float *b, int n) {
#define element(_x, _y) (*(a + ((_y) * (n) + (_x))))
	unsigned int i, j;
	for (j = 0; j < n; j++) {
		printf("[");
		for (i = 0; i < n; i++)
			printf("%6.3f ", element(i, j));
		printf("| %6.3f ]\n", b[j]);
	}
#undef element
}
