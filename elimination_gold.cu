#include "elimination_gold.h"

void elimination_gold(float *a, float *b, int n) {
#define element(_x, _y) (*(a + ((_y) * (n) + (_x))))
#define printa(s) printf("%s",s);for(j=0;j<n;j++){for(i=0;i<n;i++){printf("%f ",element(i,j));}printf("\n");}
	unsigned int i, j;
	unsigned int xx, yy, rr;
	float c;

	for (yy = 0; yy < n; yy++) {
		float pivot = element(yy, yy);

		for (xx = 0; xx < n; xx++)
			element(xx, yy) /= pivot;
		b[yy] /= pivot;

		for (rr = 0; rr < n; rr++) {
			if (rr != yy) {
				c = element(yy, rr);
				for (xx = 0; xx < n; xx++)
					element(xx, rr) -= c * element(xx, yy);
				b[rr] -= c * b[yy];
			}
		}
	}

	printa("asdf\n");

	/*
	yy = 0;
	float pivot = element(yy, yy);
	for (xx = 0; xx < n; xx++)
		element(xx, yy) /= pivot;
	b[yy] /= pivot;

	rr = 1;
	c = element(yy, rr);
	for (xx = 0; xx < n; xx++)
		element(xx, rr) -= c * element(xx, yy);
	b[rr] -= c * b[yy];

	rr = 2;
	c = element(yy, rr);
	for (xx = 0; xx < n; xx++)
		element(xx, rr) -= c * element(xx, yy);
	b[rr] -= c * b[yy];

	printa("after\n");
	*/
	/*
	for (yy = 0; yy < n; yy++) {
		float pivot = element(yy, yy);

		if (pivot != 1) {
			for (xx = 0; xx < n; xx++)
				element(xx, yy) /= pivot;
			b[yy] /= pivot;
		}

		for (rr = 0; rr < n; rr++) {
			if (rr != yy) {
				//for (xx = 0; xx < n; xx++)
					//element(xx, rr) -= element(rr, yy) * element(xx, yy);
					//element(xx, yy) -= element(yy, rr) * element(xx, yy);
					//element(xx, rr) -= element(xx, yy) * element(rr, yy);
			}
		}
	}
	*/
/*
	unsigned int xx, yy, rr, i, j;
	printa("a before:\n");
	for (yy = 0; yy < n; yy++) {
		// Locate the pivot element
		float pivot = element(yy, yy);

		// Divide all row elements by pivot element
		for (xx = 0; xx < n; xx++)
			element(xx, yy) /= pivot;
		b[yy] /= pivot;

		// Subtract rows to introduce zeros

		for (rr = 0; rr < n; rr++) {
			if (yy != rr) {
				float c = element(yy, rr);
				for (xx = 0; xx < n; xx++)
					element(xx, rr) -= c * element(xx, yy);
				b[rr] -= c * b[yy];
			}
		}
	}
	printa("a after:\n");
*/
}
