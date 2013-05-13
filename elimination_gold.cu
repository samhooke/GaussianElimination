#include "elimination_gold.h"

void elimination_gold(float *a, float *b, int n) {
#define element(_x, _y) (*(a + ((_y) * (n) + (_x))))
	unsigned int xx, yy, rr;
	unsigned int i, j;
	float c;

	for (yy = 0; yy < n; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1
		for (xx = 0; xx < n; xx++)
			element(xx, yy) /= pivot;
		b[yy] /= pivot;

		// Print out matrix
		printf("Matrix (Stage 1; Column %d):\n", yy);
		for (j = 0; j < n; j++) {
			printf("[");
			for (i = 0; i < n; i++) {
				printf("%6.3f ", element(i, j));
			}
			printf("| %6.3f ]\n", b[j]);
		}

		// Make all other values in the pivot column be zero
		for (rr = 0; rr < n; rr++) {
			if (rr != yy) {
				c = element(yy, rr);
				for (xx = 0; xx < n; xx++)
					element(xx, rr) -= c * element(xx, yy);
				b[rr] -= c * b[yy];
			}
		}

		// Print out matrix
		printf("Matrix (Stage 2; Column %d):\n", yy);
		for (j = 0; j < n; j++) {
			printf("[");
			for (i = 0; i < n; i++) {
				printf("%6.3f ", element(i, j));
			}
			printf("| %6.3f ]\n", b[j]);
		}
	}
#undef element
}
