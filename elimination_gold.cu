#include "elimination_gold.h"

void elimination_gold(SquareMatrix M, SquareMatrix N) {

	if (M.size != N.size) {
		printf("Dimensions of M and N do not match\n");
		return;
	}

	for (unsigned int i = 0; i < M.size * M.size; i++) {
		N.elements[i] = M.elements[i];
	}

}
