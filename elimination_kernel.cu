#include "elimination_kernel.h"

float elimination_kernel(float *a, float *b, int size, int kernel) {
	// Start timers
	check("Creating timers");
	cudaEvent_t timer1, timer2;
	cudaEventCreate(&timer1);
	cudaEventCreate(&timer2);
	cudaEventRecord(timer1, 0);

	// Copy data to GPU
	int sizeTotal = (size + 1) * size;
	float *g_a, *g_b;
	check("Allocating memory");
	cudaMalloc((void**)&g_a, sizeTotal * sizeof(float));
	cudaMalloc((void**)&g_b, sizeTotal * sizeof(float));
	check("Copying memory from host to device");
	cudaMemcpy(g_a, a, sizeTotal * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(1,1,1);
	dim3 dimGrid(1,1,1);

	// Execute kernel on GPU
	check("Executing kernel on GPU");
	switch (kernel) {
	case 0:
		elimination0<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 1:
		dimBlock.x = size + 1;
		elimination1<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 2:
		dimBlock.x = size + 1;
		dimBlock.y = size;
		elimination2<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	}

	// Copy data from GPU
	check("Copying data from device to host");
	cudaMemcpy(b, g_b, sizeTotal * sizeof(float), cudaMemcpyDeviceToHost);

	// Tidy up
	check("Freeing memory");
	cudaFree(g_a);
	cudaFree(g_b);

	// Stop timers
	cudaEventRecord(timer2, 0);
	cudaEventSynchronize(timer1);
	cudaEventSynchronize(timer2);
	float elapsed;
	cudaEventElapsedTime(&elapsed, timer1, timer2);

	cudaDeviceReset();
	return elapsed;
}

// Naive implementation, identical to CPU code
__global__ void elimination0(float *a, float *b, int size) {
#define element(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;
	float c;

	// The matrix will be modified in place, so first make a copy of matrix a
	for (unsigned int i = 0; i < (size + 1) * size; i++)
		b[i] = a[i];

	for (yy = 0; yy < size; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1
		for (xx = 0; xx < size + 1; xx++)
			element(xx, yy) /= pivot;

		// Make all other values in the pivot column be zero
		for (rr = 0; rr < size; rr++) {
			if (rr != yy) {
				c = element(yy, rr);
				for (xx = 0; xx < size + 1; xx++)
					element(xx, rr) -= c * element(xx, yy);
			}
		}
	}
#undef element
}

// Inner xx loops are now parallel
// Uses one block, so limited by max thread per block limit
// Still uses only global memory
__global__ void elimination1(float *a, float *b, int size) {
#define element(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;

	// The matrix will be modified in place, so first make a copy of matrix a
	for (unsigned int i = 0; i < (size + 1) * size; i++)
		b[i] = a[i];

	xx = threadIdx.x;

	for (yy = 0; yy < size; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1
		element(xx, yy) /= pivot;

		// Make all other values in the pivot column be zero
		for (rr = 0; rr < size; rr++) {
			if (rr != yy)
				element(xx, rr) -= element(yy, rr) * element(xx, yy);
		}
	}
#undef element
}

// Referenced from: http://www.cs.rutgers.edu/~venugopa/parallel_summer2012/ge.html
__global__ void elimination2(float *a, float *b, int n) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	// Allocate memory in the shared memory of the device
	__shared__ float temp[16][16];

	// Copy the data to the shared memory
	temp[idy][idx] = a[(idy * (n+1)) + idx];

	for (unsigned int i = 1; i < n; i++) {
		// No thread divergence occurs
		if ((idy + i) < n) {
			float c = (-1) * (temp[i - 1][i - 1] / temp[i + idy][i - 1]);
			temp[i + idy][idx] = temp[i - 1][idx] + ((c) * (temp[i + idy][idx]));
		}
		__syncthreads();
	}
	b[idy * (n + 1) + idx] = temp[idy][idx];
}
