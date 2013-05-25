#include <stdio.h>
#include "matrix.h"
#include "elimination_cpu.h"
#include "elimination_gpu.h"
#include "check.h"

void enter();

int main() {

	// ============================ Kernel =========================== //

	//@@ Select CPU and GPU kernel
	int kernel_cpu = 1;
	int kernel_gpu = 1;

	// ============================ Matrix =========================== //

	//@@ Size of input matrix
	// Some GPU kernels work only with specific sizes
	int size = 511;

	//@@ Type of input matrix
	// -1 = generate matrix filled with random non-zero values
	//  0 = generate matrix filled with zeros
	// >1 = read matrix from file
	int type = -1;

	// ======================= Test & Statistics ======================= //

	//@@ How many times to test each kernel
	int test_num_cpu = 0;
	int test_num_gpu = 10;

	//@@ Whether to show statistics at the end
	bool show_end_statistics = true;



	check("Generating input matrix m_in");
	float* m_in = matrix_generate(size, type);
	check("Generating blank output matrix m_out_cpu");
	float* m_out_cpu = matrix_generate(size, 0);
	check("Generating blank output matrix m_out_gpu");
	float* m_out_gpu = matrix_generate(size, 0);

	// Timers
	float elapsed_cpu = 0;
	float elapsed_gpu = 0;

	// Perform Gaussian Elimination
	float t;
	int i;
	if (test_num_cpu > 0) {
		printf(" Execution Time (ms): \n");
		printf("---------------------\n");
		for (i = 0; i < test_num_cpu; i++) {
			check("Performing Gaussian Elimination on CPU");
			t = elimination_cpu(m_in, m_out_cpu, size, kernel_cpu);
			printf("%f\n", t);
			elapsed_cpu += t;
		}
		printf("\n");
		elapsed_cpu /= test_num_cpu;
	}

	if (test_num_gpu > 0) {
		printf(" Execution Time (ms): \n");
		printf("---------------------\n");
		for (i = 0; i < test_num_gpu; i++) {
			check("Performing Gaussian Elimination on GPU");
			t = elimination_gpu(m_in, m_out_gpu, size, kernel_gpu);
			printf("%f\n", t);
			elapsed_gpu += t;
		}
		printf("\n");
		elapsed_gpu /= test_num_gpu;
	}

	if (show_end_statistics) {
		printf("Computation finished. Statistics follow:\n");
		printf("CPU average: %fms\n", elapsed_cpu);
		printf("GPU average: %fms\n", elapsed_gpu);

		// Compare the results with a threshold of tolerance
		float tolerance;
		for (tolerance = 10000.0f; tolerance > 0.00001f; tolerance /= 10) {
			float match_b = matrix_compare_b(m_out_cpu, m_out_gpu, size, tolerance);
			printf("%6.2f%% match at %8.4f tolerance\n", match_b * 100, tolerance);
		}

		float p;
		if (elapsed_gpu > 0)
			p = elapsed_cpu / elapsed_gpu;
		else
			p = -1;

		printf("GPU was %2.2f%% %s\n", ((p < 1 ? 1 / p : p) - 1) * 100, p < 1 ? "slower" : "faster");

		printf("Press enter for column 'b' results...\n");
		enter();

		printf("             CPU | GPU             \n");
		printf("-----------------+-----------------\n");
		for (i = 0; i < size; i++) {
			printf("%16.4f | %-16.4f\n", m_out_cpu[i * (size + 1) + size], m_out_gpu[i * (size + 1) + size]);
		}

		printf("Press enter for full results...\n");
		enter();

		printf("CPU results:\n");
		matrix_print(m_out_cpu, size);
		printf("GPU results:\n");
		matrix_print(m_out_gpu, size);
		printf("Original matrix:\n");
		matrix_print(m_in, size);
	}
	return 0;
}

void enter() {
	char enter = 0;
	while (enter != '\r' && enter != '\n')
		enter = getchar();
}
