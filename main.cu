#include <stdio.h>
#include "matrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"
#include "check.h"

void enter();

int main() {
	// Select GPU kernel
	int kernel = 22;

	// Whether to show statistics
	bool show_statistics = true;

	// How many times to run both algorithms
	int test_num = 1;

	// Timers
	float elapsed_cpu = 0;
	float elapsed_gpu = 0;

	// Create two identical input matrices, and two blank output matrices
	int size = 255;
	int type = -1;
	check("Generating input matrix m_in");
	float* m_in = matrix_generate(size, type);
	check("Generating blank output matrix m_out_cpu");
	float* m_out_cpu = matrix_generate(size, 0);
	check("Generating blank output matrix m_out_gpu");
	float* m_out_gpu = matrix_generate(size, 0);

	for (int i = 0; i < test_num; i++) {
		// Perform Gaussian Elimination
		check("Performing Gaussian Elimination on CPU");
		elapsed_cpu += elimination_gold2(m_in, m_out_cpu, size);
		check("Performing Gaussian Elimination on GPU");
		elapsed_gpu += elimination_kernel(m_in, m_out_gpu, size, kernel);
		check("Finished Gaussian Elimination on GPU");
	}

	elapsed_cpu /= test_num;
	elapsed_gpu /= test_num;

	if (show_statistics) {
		printf("\nComputation finished. Statistics follow:\n");
		printf("CPU (%fms)\n", elapsed_cpu);
		printf("GPU (%fms)\n", elapsed_gpu);

		// Compare the results with a threshold of tolerance
		float tolerance;
		for (tolerance = 10000.0f; tolerance > 0.00001f; tolerance /= 10) {
			float match_b = matrix_compare_b(m_out_cpu, m_out_gpu, size, tolerance);
			printf("%6.2f%% match at %8.4f tolerance\n", match_b * 100, tolerance);
		}

		float p = elapsed_cpu / elapsed_gpu;
		printf("GPU was %2.2f%% %s\n", ((p < 1 ? 1 / p : p) - 1) * 100, p < 1 ? "slower" : "faster");

		printf("Press enter for column 'b' results...\n");
		enter();

		printf("             CPU | GPU             \n");
		printf("-----------------+-----------------\n");
		for (int i = 0; i < size; i++) {
			printf("%16.4f | %-16.4f\n", m_out_cpu[i * (size + 1) + size], m_out_gpu[i * (size + 1) + size]);
		}

		printf("Press enter for full results...\n");
		enter();

		printf("CPU results:\n");
		elimination_gold_print_matrix(m_out_cpu, size);
		printf("GPU results:\n");
		elimination_gold_print_matrix(m_out_gpu, size);
		printf("Original matrix:\n");
		elimination_gold_print_matrix(m_in, size);
	}
	return 0;
}

void enter() {
	char enter = 0;
	while (enter != '\r' && enter != '\n')
		enter = getchar();
}
