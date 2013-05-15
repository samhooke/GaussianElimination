#include <stdio.h>
#include "matrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"
#include "check.h"

int main() {
	// Select GPU kernel
	int kernel = 5;

	// Timers
	float elapsed_cpu = 0;
	float elapsed_gpu = 0;

	// Create two identical input matrices, and two blank output matrices
	int size = 15;
	int type = -1;
	check("Generating input matrix m_in");
	Matrix m_in = matrix_generate(size, type);
	check("Generating blank output matrix m_out_cpu");
	Matrix m_out_cpu = matrix_generate(size, 0);
	check("Generating blank output matrix m_out_gpu");
	Matrix m_out_gpu = matrix_generate(size, 0);

	// Perform Gaussian Elimination
	check("Performing Gaussian Elimination on CPU");
	elapsed_cpu = elimination_gold(m_in.elements, m_out_cpu.elements, size);
	check("Performing Gaussian Elimination on GPU");
	elapsed_gpu = elimination_gold2(m_in.elements, m_out_gpu.elements, size);
	//elapsed_gpu = elimination_kernel(m_in.elements, m_out_gpu.elements, size, kernel);

	printf("CPU (%fms)\n", elapsed_cpu);
	elimination_gold_print_matrix(m_out_cpu.elements, size);
	printf("GPU (%fms)\n", elapsed_gpu);
	elimination_gold_print_matrix(m_out_gpu.elements, size);

	// Compare the results with a threshold of tolerance
	bool match_b = matrix_compare_b(m_out_cpu.elements, m_out_gpu.elements, size, 0.001f);

	// Show statistics
	printf("Results %s\n", match_b ? "match!" : "do not match.");
	float p = elapsed_cpu / elapsed_gpu;
	printf("GPU was %2.2f%% %s\n", (p < 1 ? 1 / p : p) * 100, p < 1 ? "slower" : "faster");

	return 0;
}
