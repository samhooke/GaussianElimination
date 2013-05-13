#include <stdio.h>
#include <stdlib.h>
#include "elimination_gold.h"
#include "elimination_kernel.h"

typedef struct {
	float* a;
	float* b;
	int size;
} Matrix;

Matrix matrix_generate(int size, int type);

int main() {
	Matrix m = matrix_generate(3, 1);

	elimination_gold(m.a, m.b, m.size);

	for (unsigned int i = 0; i < m.size; i++)
		printf("%f\n", m.b[i]);

	return 0;
}

Matrix matrix_generate(int size, int type) {
	float a[1024];
	float b[32];
	int n;

	if (type > 0) {
		// Type is > 0: Load matrix from file
		int* d = (int*) malloc(4096 * sizeof(int));
		FILE* fp;

		// Choose which file to read from
		bool flag = false;
		switch (size) {
		case 3:
			switch (type) {
			case 1:
				fp = fopen("example_3x3_1.txt", "r");
				break;
			case 2:
				fp = fopen("example_3x3_2.txt", "r");
				break;
			case 3:
				fp = fopen("example_3x3_3.txt", "r");
				break;
			default:
				flag = true;
				break;
			}
			break;
		case 6:
			switch (type) {
			case 1:
				fp = fopen("example_6x6_1.txt", "r");
				break;
			default:
				flag = true;
				break;
			}
			break;
		default:
			flag = true;
			break;
		}

		if (flag) {
			printf("Matrix of size %d and type %d does not exist.\n", size, type);
			exit(0);
		}

		int r;
		if (fscanf(fp, "%d\n", &r) == 1) {
			n = r;
		} else {
			printf("Could not read matrix size.\n");
			exit(0);
		}

		unsigned int i = 0;
		float f;
		while (fscanf(fp, "%f\n", &f) == 1 && i < ((n * n) + n)) {
			if (i < (n * n))
				a[i++] = f;
			else
				b[i++ - (n * n)] = f;
		}

		fclose(fp);
	} else {
		// Type is 0: Generate random matrix of size 'size'
		n = size;

		// Populate arrays with values between -5 and 5
		// Distribution is not uniform
		unsigned int i = 0;
		for (i = 0; i < n * n; i++)
			a[i] = (rand() % 10) - 5;
		for (i = 0; i < n; i++)
			b[i] = (rand() % 10) - 5;
	}

	// Copy generated matrix into a Matrix struct within allocated space
	Matrix m;
	m.a = (float*) malloc(n * n * sizeof(float));
	m.b = (float*) malloc(n * sizeof(float));
	m.size = n;

	int i;
	for (i = 0; i < n * n; i++)
		*(m.a + i) = (float) a[i];

	for (i = 0; i < n; i++)
		*(m.b + i) = (float) b[i];

	return m;
}
