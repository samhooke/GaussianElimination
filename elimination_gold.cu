#include "elimination_gold.h"

void elimination_gold(float *a, float *b, int n) {
#define element(_x, _y) (*(a + ((_y) * (n) + (_x))))
#define printa(s) printf("%s",s);for(j=0;j<n;j++){for(i=0;i<n;i++){printf("%f ",element(i,j));}printf("\n");}
	unsigned int xx, yy, rr, i, j;
	printa("a before:\n");

	for (yy = 0; yy < n; yy++) {
		// Locate the pivot element
		float pivot = element(yy, yy);

		// Divide all row elements by pivot element
		for (xx = 0; xx < n; xx++)
			element(xx, yy) /= pivot;
		b[yy] /= pivot;

		for (rr = 0; rr < n; rr++) {
			if (yy != rr) {
				float c = element(yy, rr);
				for (xx = 0; xx < n; xx++)
					element(xx, rr) -= c * element(xx, yy);
				b[rr] -= c * b[yy];
			}
		}

		/*
		// Make all other values in this column be zero
		for (yy2 = 0; yy2 < n; yy2++) {
			if (yy2 != yy) {
				float c = element(yy, yy2);
				for (xx = 0; xx < n; xx++)
					element(xx, yy2) -= c * element(xx, yy);
				b[yy2] -= c * b[yy];
			}
		}
		*/
		/*
		for (xx = 0; xx < n; xx++) {
			if (xx != yy) {
				for (yy2 = 0; yy2 < n; yy2++) {
					element(xx, yy2) = element(xx, yy2) - element(yy, yy2) * element(xx, yy);;
				}
				//b[yy2] = b[yy2] - element(yy, yy2) * element(xx, yy);
			}
		}
		*/
	}

	printa("a after:\n");
}
