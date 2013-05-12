#include "elimination_gold.h"

void elimination_gold(SquareMatrix M, SquareMatrix N) {

	// Verify dimensions match
	if (M.size != N.size) {
		printf("Dimensions of M and N do not match\n");
		return;
	}

	// Check if M is all zeros
	bool zero = true;
	for (unsigned int i = 0; i < M.size * M.size; i++) {
		if (M.elements[i] != 0) {
			zero = false;
			break;
		}
	}

	// (3) If M is all zeros, simply return M
	if (zero) {
		for (unsigned int i = 0; i < M.size * M.size; i++) {
			N.elements[i] = M.elements[i];
		}
		return;
	}

	/*
	// Loop through columns, then rows
	unsigned int c;
	unsigned int r;
	for (c = 0; c < M.size; c++) {
		// (4) (5) Check if column is nonzero, and if so, find first nonzero row in the column
		bool nonzero = false;
		for (r = 0; r < M.size; r++) {
			if (M.elements[r * M.size + c] != 0) {
				nonzero = true;
				break;
			}
		}

		if (nonzero) {
			// (6) (7)
			unsigned int ru = r * (1 / M.elements[r * M.size + c]);

			// (8)
			if (r != )
		}
	}
	*/
}
