#include <stdio.h>
#include "matrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"

void handleError(char* location);

int main() {
	// Select GPU kernel
	int kernel = 1;

	// Timers
	float elapsed_cpu = 0;
	float elapsed_gpu = 0;

	// Create two copies of the same matrix
	Matrix m = matrix_generate(6, 1);
	Matrix n = matrix_generate(6, 1);

	// Perform Gaussian Elimination

	// TODO:
	// Move timer code inside the elimination_*() functions
	// Have them return the time taken through a return
	// Change the *.a, *.b, *.size to instead passing the Matrix itself
	elapsed_cpu = elimination_gold(m.a, m.b, m.size);
	elapsed_gpu = elimination_kernel(n.a, n.b, n.size, kernel);

	printf("CPU (%fms):\n", elapsed_cpu);
	elimination_gold_print_matrix(m.a, m.b, m.size);
	printf("GPU (%fms):\n", elapsed_gpu);
	elimination_gold_print_matrix(n.a, n.b, n.size);

	// Compare the results with a threshold of tolerance
	bool match_b = matrix_compare_b(m.b, n.b, m.size, 0.00001f);

	// Show statistics
	printf("Results %s\n", match_b ? "match!" : "do not match.");
	//for (unsigned int i = 0; i < m.size; i++)
	//	printf("x%-3d = %10.6f\n", i, m.b[i]);

	return 0;
}

void handleError(char* location) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: (%s) %s\n", location, cudaGetErrorString(err));
	}
}
