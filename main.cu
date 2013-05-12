#include <stdio.h>
#include <stdlib.h>
#include "squarematrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"



int main() {
	/*
	// Matrix size
	#define N 6
	float a[] = {
			1.00, 0.00, 0.00, 0.00, 0.00, 0.00,
			1.00, 0.63, 0.39, 0.25, 0.16, 0.10,
			1.00, 1.26, 1.58, 1.98, 2.49, 3.13,
			1.00, 1.88, 3.55, 6.70, 6.62, 3.80,
			1.00, 2.51, 6.32, 5.88, 9.90, 0.28,
			1.00, 3.14, 9.87, 3.01, 9.41, 6.02
	};
	float b[] = {
			0.01,
			0.61,
			0.91,
			0.99,
			0.60,
			0.02
	};
*/
	/*
	// Matrix size
	#define N 3
	float a[] = {
			1, 2, 3,
			4, 5, 6,
			7, 8, 9
	};
	float b[] = {
			1,
			2,
			3
	};
	*/
	// Matrix size
	#define N 3
	float a[] = {
			-5, 2, 1,
			1, -8, 3,
			3, 1, 7
	};
	float b[] = {
			2,
			-6,
			-16
	};

	elimination_gold(a, b, N);

	for (unsigned int i = 0; i < N; i++)
		printf("%f\n", b[i]);

	return 0;
}
