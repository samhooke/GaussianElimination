#include <stdio.h>
#include "matrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"
#include "check.h"

int main() {
	// Select GPU kernel
	int kernel = 2;

	// Timers
	float elapsed_cpu = 0;
	float elapsed_gpu = 0;

	// Create two copies of the same matrix
	int size = 3;
	int type = 1;
	check("Generating matrix m");
	Matrix m = matrix_generate(size, type);
	check("Generating matrix n");
	Matrix n = matrix_generate(size, type);

	printf("Very much first matrix:\n");
	elimination_gold_print_matrix(m.elements, m.size);

	// Perform Gaussian Elimination
	check("Performing Gaussian Elimination on CPU");
	elapsed_cpu = elimination_gold(m.elements, m.size);
	check("Performing Gaussian Elimination on GPU");
	//elapsed_gpu = elimination_kernel(n.elements, n.size, kernel);
	elapsed_cpu = elimination_gold(n.elements, n.size);

	printf("CPU (%fms)\n", elapsed_cpu);
	elimination_gold_print_matrix(m.elements, m.size);
	printf("GPU (%fms)\n", elapsed_gpu);
	elimination_gold_print_matrix(n.elements, n.size);

	// Compare the results with a threshold of tolerance
	bool match_b = matrix_compare_b(m.elements, n.elements, m.size, 0.001f);

	// Show statistics
	printf("Results %s\n", match_b ? "match!" : "do not match.");
	float p = elapsed_cpu / elapsed_gpu;
	printf("GPU was %2.2f%% %s\n", (p < 1 ? 1 / p : p) * 100, p < 1 ? "slower" : "faster");

	return 0;
}
