#include <stdio.h>
#include "matrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"

int main() {
	// Create two copies of the same matrix
	Matrix m = matrix_generate(3, 1);
	Matrix n = m;

	// Perform Gaussian Elimination
	elimination_gold(m.a, m.b, m.size);
	elimination_gold(n.a, n.b, n.size);

	// Compare the results with a threshold of tolerance
	bool match_b = matrix_compare_b(m.b, n.b, m.size, 0.00001f);

	// Show statistics
	printf("%s", match_b ? "Results match!\n" : "Results do not match.\n");
	for (unsigned int i = 0; i < m.size; i++)
		printf("x%-3d = %10.6f\n", i, m.b[i]);


	return 0;
}
