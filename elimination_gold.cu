#include "elimination_gold.h"
#include <stdio.h>

// Performs Gauss-Jordan elimination on the CPU; [A]{x]={b}
// Inputs:
//   a -> [A], the matrix of coefficients of size 'n' by 'n'
//   b -> {b}, the vertical matrix of results
//   n -> width/height of 'a', and height of 'b'
// Outputs:
//   Modifies 'a' into the identity matrix
//   Modifies 'b' into the solution for {x}
float elimination_gold(float *a, float *b, int size) {
	// Start timers
	cudaEvent_t timer1, timer2;
	cudaEventCreate(&timer1);
	cudaEventCreate(&timer2);
	cudaEventRecord(timer1, 0);

#define element(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;
	float c;

	// The matrix will be modified in place, so first make a copy of matrix a
	for (unsigned int i = 0; i < (size + 1) * size; i++)
		b[i] = a[i];

#ifdef DEBUG
		printf("Matrix before:\n");
		elimination_gold_print_matrix(b, size);
#endif

	for (yy = 0; yy < size; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1
		for (xx = 0; xx < size + 1; xx++)
			element(xx, yy) /= pivot;

#ifdef DEBUG
		printf("Matrix (Stage 1; Column %d):\n", yy);
		elimination_gold_print_matrix(b, size);
#endif

		// Make all other values in the pivot column be zero
		for (rr = 0; rr < size; rr++) {
			if (rr != yy) {
				c = element(yy, rr);
				for (xx = 0; xx < size + 1; xx++)
					element(xx, rr) -= c * element(xx, yy);
			}
		}

#ifdef DEBUG
		printf("Matrix (Stage 2; Column %d):\n", yy);
		elimination_gold_print_matrix(b, size);
#endif

	}
#undef element

	// Stop timers
	cudaEventRecord(timer2, 0);
	cudaEventSynchronize(timer1);
	cudaEventSynchronize(timer2);
	float elapsed;
	cudaEventElapsedTime(&elapsed, timer1, timer2);
	return elapsed;
}

float elimination_gold2(float *a, float *b, int size) {
	// Start timers
	cudaEvent_t timer1, timer2;
	cudaEventCreate(&timer1);
	cudaEventCreate(&timer2);
	cudaEventRecord(timer1, 0);

#define element(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;
	float c;

	// The matrix will be modified in place, so first make a copy of matrix a
	for (unsigned int i = 0; i < (size + 1) * size; i++)
		b[i] = a[i];

#ifdef DEBUG
		printf("Matrix before:\n");
		elimination_gold_print_matrix(b, size);
#endif

	for (yy = 0; yy < size; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1

		// We know that pivot / pivot will equal 1, so just set it to 1
		// This line can be commented out, and the final column will still be correct
		element(yy, yy) = 1;

		// Start from yy + 1 instead of 0. The + 1 is because we have calculated done the pivot
		for (xx = yy + 1; xx < size + 1; xx++)
			element(xx, yy) /= pivot;

#ifdef DEBUG
		printf("Matrix (Stage 1; Column %d):\n", yy);
		elimination_gold_print_matrix(b, size);
#endif

		// Make all other values in the pivot column be zero
		for (rr = 0; rr < size; rr++) {
			if (rr != yy) {
				c = element(yy, rr);

				// We know that this value will be zero
				// This line can be commented out, and the final column will still be correct
				element(yy, rr) = 0;

				// Start from yy + 1 instead of 0. The + 1 is because we have already set one value to zero
				for (xx = yy + 1; xx < size + 1; xx++)
					element(xx, rr) -= c * element(xx, yy);
			}
		}

#ifdef DEBUG
		printf("Matrix (Stage 2; Column %d):\n", yy);
		elimination_gold_print_matrix(b, size);
#endif

	}
#undef element

	// Stop timers
	cudaEventRecord(timer2, 0);
	cudaEventSynchronize(timer1);
	cudaEventSynchronize(timer2);
	float elapsed;
	cudaEventElapsedTime(&elapsed, timer1, timer2);
	return elapsed;
}

// This method suffers some loss in precision and is also slower
// However, the main loop is simpler
float elimination_gold3(float *a, float *b, int size) {
	// Start timers
	cudaEvent_t timer1, timer2;
	cudaEventCreate(&timer1);
	cudaEventCreate(&timer2);
	cudaEventRecord(timer1, 0);

#define element(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;
	float pivot, c;

	for (unsigned int i = 0; i < (size + 1) * size; i++)
		b[i] = a[i];

#ifdef DEBUG
		printf("Matrix before:\n");
		elimination_gold_print_matrix(b, size);
#endif

	for (yy = 0; yy < size; yy++) {
		pivot = element(yy, yy);

		for (rr = 0; rr < size; rr++) {
			if (rr != yy) {
				c = element(yy, rr);

				// Combine the subtracting and dividing into one operation
				for (xx = yy + 1; xx < size + 1; xx++)
					element(xx, rr) -= c * element(xx, yy) / pivot;
			}

#ifdef DEBUG
		printf("Matrix (Column %d):\n", yy);
		elimination_gold_print_matrix(b, size);
#endif

		}
	}

	// However, one final division is still required for the last column
	for (yy = 0; yy < size; yy++) {
		element(size, yy) /= element(yy, yy);
	}
#undef element

	// Stop timers
	cudaEventRecord(timer2, 0);
	cudaEventSynchronize(timer1);
	cudaEventSynchronize(timer2);
	float elapsed;
	cudaEventElapsedTime(&elapsed, timer1, timer2);
	return elapsed;
}

// Prints a matrix in the format of [A]{b}
// Inputs:
//   a -> [A], the matrix of coefficients of size 'n' by 'n'
//   b -> {b}, the vertical matrix of results
//   n -> width/height of 'a', and height of 'b'
// Outputs:
//   Prints out the matrix as a nicely formatted table
void elimination_gold_print_matrix(float *elements, int size) {
	bool front, end;

	for (unsigned int i = 0; i < (size + 1) * size; i++) {
		front = (i % (size + 1) == 0);
		end = (i % (size + 1) == size );

		if (front) printf("[ ");
		if (end) printf("| ");
		printf("%8.4f ", *(elements + i));
		if (end) printf("]\n");
	}
}
