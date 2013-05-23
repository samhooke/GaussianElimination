#include "matrix.h"

#define MAX_SIZE 4096
#define MAX_SIZE_TOTAL (MAX_SIZE + 1) * MAX_SIZE
#define MAX_FILE_READ_SIZE MAX_SIZE * 10 // Some arbitrary number larger than MAX_SIZE

// Generates a matrix of dimensions size by size
// If type == -1, the matrix is filled with random values
// If type ==  0, the matrix is filled with zeros
// If type >=  1, the matrix is loaded from file
float* matrix_generate(int size, int type) {

	int sizeTotal = (size + 1) * size;

	if (sizeTotal > MAX_SIZE_TOTAL) {
		fprintf(stderr, "Requested size %d (%dx%d) bigger than max size %d.\n", sizeTotal, size + 1, size, MAX_SIZE_TOTAL);
		exit(0);
	}

	float *a = (float*) malloc(MAX_SIZE_TOTAL * sizeof(float));

	if (type > 0) {
		// Type is > 0: Load matrix from file
		int* d = (int*) malloc(MAX_FILE_READ_SIZE * sizeof(int));
		FILE* fp;

		// Choose which file to read from
		bool flag = false;
		switch (size) {
		case 3:
			switch (type) {
			case 1:
				fp = fopen("data/3x3_1.txt", "r");
				break;
			case 2:
				fp = fopen("data/3x3_2.txt", "r");
				break;
			case 3:
				fp = fopen("data/3x3_3.txt", "r");
				break;
			default:
				flag = true;
				break;
			}
			break;
		case 6:
			switch (type) {
			case 1:
				fp = fopen("data/6x6_1.txt", "r");
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
			fprintf(stderr, "Matrix of size %d and type %d does not exist.\n", size, type);
			exit(0);
		}

		int r;
		if (fscanf(fp, "%d\n", &r) == 1) {
			if (size != r) {
				fprintf(stderr, "Requested matrix size was %d, but size of matrix in file says it is %d.\n", size, r);
				exit(0);
			}
		} else {
			fprintf(stderr, "Could not read matrix size.\n");
			exit(0);
		}

		unsigned int i = 0;
		float f;
		while (fscanf(fp, "%f\n", &f) == 1 && i < sizeTotal)
			a[i++] = f;

		fclose(fp);
	} else if (type == -1) {
		// Type is -1: Generate random matrix of size 'size'

		// Use same seed to ensure identical matrix
		srand(123);

		// Populate arrays with values between about -10000 and 10000
		// Distribution is not uniform
		for (unsigned int i = 0; i < sizeTotal; i++) {
			do {
				a[i] = (rand() % 20000) - 10000;
			} while ((int) a[i] == 0);
		}
	} else if (type == 0) {
		// Type is 0: Generate a blank matrix

		for (unsigned int i = 0; i < sizeTotal; i++)
			a[i] = 0;
	} else {
		fprintf(stderr, "Invalid type: %d\n", type);
		exit(0);
	}

	// Copy generated matrix into a Matrix struct within allocated space
	float *elements = (float*) malloc(sizeTotal * sizeof(float));

	for (unsigned int i = 0; i < sizeTotal; i++)
		*(elements + i) = (float) a[i];

	return elements;
}

float matrix_compare_b(float *m, float *n, int size, float tolerance) {
	int sizeTotal = (size + 1) * size;
	int numMatch = 0;

	// Compare only last column
	for (unsigned int i = size + 1 - 1; i < sizeTotal; i+= size + 1)
		if (m[i] < n[i] + tolerance && m[i] > n[i] - tolerance)
			numMatch++;

	return (float) numMatch / (float) size;
}
