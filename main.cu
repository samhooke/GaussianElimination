#include <stdio.h>
#include "matrix.h"
#include "elimination_gold.h"
#include "elimination_kernel.h"

int main() {
	Matrix m = matrix_generate(3, 1);

	elimination_gold(m.a, m.b, m.size);

	for (unsigned int i = 0; i < m.size; i++)
		printf("%f\n", m.b[i]);

	return 0;
}
