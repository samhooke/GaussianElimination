#include <stdio.h>
#include "matrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"
#include "check.h"

void enter();

int main() {
	// Select GPU kernel
	int kernel = 11;

	// Timers
	float elapsed_cpu = 0;
	float elapsed_gpu = 0;

	// Create two identical input matrices, and two blank output matrices
	int size = 64;
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
	//elapsed_gpu = elimination_gold2(m_in.elements, m_out_gpu.elements, size);
	elapsed_gpu = elimination_kernel(m_in.elements, m_out_gpu.elements, size, kernel);
	check("Finished Gaussian Elimination on GPU");

	printf("\nComputation finished. Statistics follow:\n");
	printf("CPU (%fms)\n", elapsed_cpu);
	printf("GPU (%fms)\n", elapsed_gpu);

	// Compare the results with a threshold of tolerance
	float tolerance;
	for (tolerance = 1.0f; tolerance > 0.00001f; tolerance /= 10) {
		float match_b = matrix_compare_b(m_out_cpu.elements, m_out_gpu.elements, size, tolerance);
		printf("%6.2f%% match at %.4f tolerance\n", match_b * 100, tolerance);
	}

	float p = elapsed_cpu / elapsed_gpu;
	printf("GPU was %2.2f%% %s\n", ((p < 1 ? 1 / p : p) - 1) * 100, p < 1 ? "slower" : "faster");

	printf("Press enter for column 'b' results...\n");
	enter();

	printf("             CPU | GPU             \n");
	printf("-----------------+-----------------\n");
	for (int i = 0; i < size; i++) {
		printf("%16.4f | %-16.4f\n", m_out_cpu.elements[i * (size + 1) + size], m_out_gpu.elements[i * (size + 1) + size]);
		//printf("%16.4f | %-16.4f\n", m_out_cpu.elements[i * (size + 1) + size], m_out_gpu.elements[i * (size + 1) + size] / m_out_gpu.elements[i * (size + 1) + i]);
	}

	printf("Press enter for full results...\n");
	enter();

	printf("CPU results:\n");
	elimination_gold_print_matrix(m_out_cpu.elements, size);
	printf("GPU results:\n");
	elimination_gold_print_matrix(m_out_gpu.elements, size);

	return 0;
}

void enter() {
	char enter = 0;
	while (enter != '\r' && enter != '\n')
		enter = getchar();
}
