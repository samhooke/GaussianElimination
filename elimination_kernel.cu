#include "elimination_kernel.h"

#define BLOCK_SIZE 16

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
	case 3:
		dimBlock.x = size + 1;
		dimBlock.y = size;
		elimination3<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 4:
		dimBlock.x = size + 1;
		dimBlock.y = size;
		elimination4<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 5:
		dimBlock.x = BLOCK_SIZE;
		dimBlock.y = BLOCK_SIZE;
		dimGrid.x = 1;
		dimGrid.y = 1;
		//dimGrid.x = (size + 1 - 1) / BLOCK_SIZE + 1;
		//dimGrid.y = (size - 1) / BLOCK_SIZE + 1;
		elimination5<<<dimGrid, dimBlock>>>(g_a, g_b, size);
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

// Very, very naive implementation, identical to CPU code
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
// Uses one block, so limited to 512 threads, so the matrix can be at most of size 22
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

// Both xx and rr loops are now in parallel
__global__ void elimination2(float *a, float *b, int size) {
#define element(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;

	// The matrix will be modified in place, so first make a copy of matrix a
	for (unsigned int i = 0; i < (size + 1) * size; i++)
		b[i] = a[i];

	__syncthreads();

	xx = threadIdx.x;
	rr = threadIdx.y;

	for (yy = 0; yy < size; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1
		element(xx, yy) /= pivot;

		// Make all other values in the pivot column be zero
		if (rr != yy)
			element(xx, rr) -= element(yy, rr) * element(xx, yy);

		__syncthreads();
	}
#undef element
}

// Data is copied in parallel
__global__ void elimination3(float *a, float *b, int size) {
#define element(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;

	xx = threadIdx.x;
	rr = threadIdx.y;

	int tid = rr * (size + 1) + xx;

	// The matrix will be modified in place, so first make a copy of matrix a
	b[tid] = a[tid];

	__syncthreads();

	for (yy = 0; yy < size; yy++) {
		float pivot = element(yy, yy);

		// Make the pivot be 1
		element(xx, yy) /= pivot;

		// Make all other values in the pivot column be zero
		if (rr != yy)
			element(xx, rr) -= element(yy, rr) * element(xx, yy);

		__syncthreads();
	}
#undef element
}

// Shared memory is used
// However, still limited to matrices of size 22
__global__ void elimination4(float *a, float *b, int size) {
#define element(_x, _y) (*(sdata + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;

	// With a limit of 512 threads per block, and only one block, this results in a maximum
	// of a matrix size 22, which requires (22 + 1) x 22 values
	__shared__ float sdata[(22 + 1) * 22];

	xx = threadIdx.x;
	rr = threadIdx.y;

	int tid = rr * (size + 1) + xx;

	// The matrix will be modified in place, so first make a copy of matrix a
	sdata[tid] = a[tid];

	for (yy = 0; yy < size; yy++) {

		__syncthreads();

		// Make the pivot be 1
		element(xx, yy) /= element(yy, yy);

		__syncthreads();

		// Make all other values in the pivot column be zero
		if (rr != yy)
			element(xx, rr) -= element(yy, rr) * element(xx, yy);
	}

	b[tid] = sdata[tid];
#undef element
}

__global__ void elimination5(float *a, float *b, int size) {
#define element(_x, _y) (*(sdata + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;

	__shared__ float sdata[BLOCK_SIZE * BLOCK_SIZE];

	xx = threadIdx.x;
	rr = threadIdx.y;

	int tid = rr * (size + 1) + xx;

	sdata[tid] = a[blockDim.y * BLOCK_SIZE + blockDim.x + tid];

	for (yy = 0; yy < size; yy++) {

		__syncthreads();

		// Make the pivot be 1
		element(xx, yy) /= element(yy, yy);

		__syncthreads();

		// Make all other values in the pivot column be zero
		if (rr != yy)
			element(xx, rr) -= element(yy, rr) * element(xx, yy);
	}

	b[blockDim.y * BLOCK_SIZE + blockDim.x + tid] = sdata[tid];
#undef element
}
