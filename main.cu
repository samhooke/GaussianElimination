#include <stdio.h>
#include "matrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"
#include "check.h"

int main() {
	// Select GPU kernel
	int kernel = 1;

	// Timers
	float elapsed_cpu = 0;
	float elapsed_gpu = 0;

	// Create two copies of the same matrix
	int size = 32;
	int type = 0;
	check("Generating matrix m");
	Matrix m = matrix_generate(size, type);
	check("Generating matrix n");
	Matrix n = matrix_generate(size, type);

	// Perform Gaussian Elimination
	check("Performing Gaussian Elimination on CPU");
	elapsed_cpu = elimination_gold(m.a, m.b, m.size);
	check("Performing Gaussian Elimination on GPU");
	elapsed_gpu = elimination_kernel(n.a, n.b, n.size, kernel);

	printf("CPU (%fms)\n", elapsed_cpu);
	//elimination_gold_print_matrix(m.a, m.b, m.size);
	printf("GPU (%fms)\n", elapsed_gpu);
	//elimination_gold_print_matrix(n.a, n.b, n.size);

	// Compare the results with a threshold of tolerance
	bool match_b = matrix_compare_b(m.b, n.b, m.size, 0.001f);

	// Show statistics
	printf("Results %s\n", match_b ? "match!" : "do not match.");

	return 0;
}
