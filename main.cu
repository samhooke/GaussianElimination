#include <stdio.h>
#include "matrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"

int main() {
	// Create two copies of the same matrix
	Matrix m = matrix_generate(3, 1);
	Matrix n = matrix_generate(3, 1);

	// Perform Gaussian Elimination
	int kernel = 0;
	elimination_gold(m.a, m.b, m.size);
	elimination_kernel(n.a, n.b, n.size, kernel);

	printf("Matrix m:\n");
	elimination_gold_print_matrix(m.a, m.b, m.size);
	printf("Matrix n:\n");
	elimination_gold_print_matrix(n.a, n.b, n.size);

	// Compare the results with a threshold of tolerance
	bool match_b = matrix_compare_b(m.b, n.b, m.size, 0.00001f);

	// Show statistics
	printf("Results %s\n", match_b ? "match!" : "do not match.");
	for (unsigned int i = 0; i < m.size; i++)
		printf("x%-3d = %10.6f\n", i, m.b[i]);

	return 0;
}
