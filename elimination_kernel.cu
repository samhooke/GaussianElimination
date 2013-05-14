#include "elimination_kernel.h"

float elimination_kernel(float *a, float *b, int n, int kernel) {
	// Start timers
	check("Creating timers");
	cudaEvent_t timer1, timer2;
	cudaEventCreate(&timer1);
	cudaEventCreate(&timer2);
	cudaEventRecord(timer1, 0);

	// Copy data to GPU
	int size_a = n * n;
	int size_b = n;
	float *g_a;
	float *g_b;
	check("Allocating memory");
	cudaMalloc((void**)&g_a, size_a * sizeof(float));
	cudaMalloc((void**)&g_b, size_b * sizeof(float));
	check("Copying memory from host to device");
	cudaMemcpy(g_a, a, size_a * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_b, b, size_b * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(1,1,1);
	dim3 dimGrid(1,1,1);

	// Execute kernel on GPU
	check("Executing kernel on GPU");
	switch (kernel) {
	case 0:
		elimination0<<<dimGrid, dimBlock>>>(g_a, g_b, n);
		break;
	case 1:
		dimBlock.x = n;
		elimination1<<<dimGrid, dimBlock>>>(g_a, g_b, n);
		break;
	case 2:
		dimBlock.x = n + 1;
		dimBlock.y = n;
		elimination2<<<dimGrid, dimBlock>>>(g_a, g_b, n);
		break;
	}

	// Copy data from GPU
	check("Copying data from device to host");
	cudaMemcpy(a, g_a, size_a * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, g_b, size_b * sizeof(float), cudaMemcpyDeviceToHost);

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

// Inner xx loops are now parallel
// Uses one block, so limited by max thread per block limit
// Still uses only global memory
__global__ void elimination1(float *a, float *b, int n) {
#define element(_x, _y) (*(a + ((_y) * (n) + (_x))))
	int xx, yy, rr;
	float c;

	xx = threadIdx.x;

	for (yy = 0; yy < n; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1
		element(xx, yy) /= pivot;
		b[yy] /= pivot;

		// Make all other values in the pivot column be zero
		for (rr = 0; rr < n; rr++) {
			if (rr != yy) {
				c = element(yy, rr);
				element(xx, rr) -= c * element(xx, yy);
				b[rr] -= c * b[yy];
			}
		}
	}
#undef element
}

// Referenced from: http://www.cs.rutgers.edu/~venugopa/parallel_summer2012/ge.html
// TODO:
// -Modify existing code to use one matrix instead of two
// -Also don't overwrite input matrix
__global__ void elimination2(float *a, float *b, int n) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	//Allocating memory in the share memory of the device
	__shared__ float temp[16][16];

	//Copying the data to the shared memory
	temp[idy][idx] = a[(idy * (n+1)) + idx] ;

	for(unsigned int i = 1; i < n ; i++) {
		// No thread divergence occurs
		if((idy + i) < n) {
			float var1 = (-1) * (temp[i - 1][i - 1] / temp[i + idy][i - 1]);
			temp[i + idy][idx] = temp[i - 1][idx] + ((var1) * (temp[i + idy][idx]));
		}
		__syncthreads(); //Synchronizing all threads before next iteration
	}
	b[idy * (n + 1) + idx] = temp[idy][idx];
}

/*
//a[threadIdx.y * n + threadIdx.x] = 3;
int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;

//extern __shared__ float s_a[];
//extern __shared__ float s_b[];

// Load into shared memory
//s_a[tx] = a[b]
*/
