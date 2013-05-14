#include "elimination_kernel.h"

void elimination_kernel(float *a, float *b, int n, int kernel) {

	// Copy data to GPU
	int size_a = n * n;
	int size_b = n;
	float *g_a;
	float *g_b;
	cudaMalloc((void**)&g_a, size_a * sizeof(float));
	cudaMalloc((void**)&g_b, size_b * sizeof(float));
	cudaMemcpy(g_a, a, size_a * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_b, b, size_b * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(1,1,1);
	dim3 dimGrid(1,1,1);

	// Execute kernel on GPU
	switch (kernel) {
	case 0:
		elimination0<<<dimGrid, dimBlock>>>(g_a, g_b, n);
		break;
	case 1:
		dimBlock.x = n;
		dimBlock.y = n;
		//elimination1<<<dimGrid, dimBlock>>>(g_a, g_b, n);
		break;
	}

	// Copy data from GPU
	cudaMemcpy(a, g_a, size_a * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, g_b, size_b * sizeof(float), cudaMemcpyDeviceToHost);

	// Tidy up
	cudaFree(g_a);
	cudaFree(g_b);
	cudaDeviceReset();
}

__global__ void elimination0(float *a, float *b, int n) {
#define element(_x, _y) (*(a + ((_y) * (n) + (_x))))
	unsigned int xx, yy, rr;
	float c;

	for (yy = 0; yy < n; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1
		for (xx = 0; xx < n; xx++)
			element(xx, yy) /= pivot;
		b[yy] /= pivot;

		// Make all other values in the pivot column be zero
		for (rr = 0; rr < n; rr++) {
			if (rr != yy) {
				c = element(yy, rr);
				for (xx = 0; xx < n; xx++)
					element(xx, rr) -= c * element(xx, yy);
				b[rr] -= c * b[yy];
			}
		}
	}
#undef element
}
