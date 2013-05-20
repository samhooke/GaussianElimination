#include "elimination_kernel.h"

// Used by kernel 5 & 11
#define BLOCK_SIZE 16

// Used by kernels 13, 14 & 15
#define ELEMENTS_PER_THREAD 4

float elimination_kernel(float *a, float *b, int size, int kernel) {
	// Start timers
	check("Creating timers");
	cudaEvent_t timer1, timer2;
	cudaEventCreate(&timer1);
	cudaEventCreate(&timer2);
	cudaEventRecord(timer1, 0);
	cudaEventSynchronize(timer1);

	int sizeTotal = (size + 1) * size;
	float *g_a, *g_b;

	if (kernel < 8) {
		// Kernels 1 to 7 require two copies of the matrix
		cudaMalloc((void**)&g_a, sizeTotal * sizeof(float));
		check("Allocated memory g_a");
		cudaMalloc((void**)&g_b, sizeTotal * sizeof(float));
		check("Allocated memory g_b");

		// Copy a to g_a which the device will use for reference
		cudaMemcpy(g_a, a, sizeTotal * sizeof(float), cudaMemcpyHostToDevice);
		check("Copied memory from host to device");
	} else {
		// Kernels 8+ require one copy of the matrix
		cudaMalloc((void**)&g_b, sizeTotal * sizeof(float));
		check("Allocated memory g_b");

		// Copy a to g_b which the device will modify in place
		cudaMemcpy(g_b, a, sizeTotal * sizeof(float), cudaMemcpyHostToDevice);
		check("Copied memory from host to device");
	}

	dim3 dimBlock(1,1,1);
	dim3 dimGrid(1,1,1);

	// Execute kernel on GPU
	switch (kernel) {
	case 0:
		// GPU Kernel 0
		elimination0<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 1:
		// GPU Kernel 1
		dimBlock.x = size + 1;
		elimination1<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 2:
		// GPU Kernel 2
		dimBlock.x = size + 1;
		dimBlock.y = size;
		elimination2<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 3:
		// GPU Kernel 3
		dimBlock.x = size + 1;
		dimBlock.y = size;
		elimination3<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 4:
		// GPU Kernel 4
		dimBlock.x = size + 1;
		dimBlock.y = size;
		elimination4<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 5:
		// GPU Kernel 5
		dimBlock.x = BLOCK_SIZE;
		dimBlock.y = BLOCK_SIZE;
		dimGrid.x = (size + 1 - 1) / BLOCK_SIZE + 1;
		dimGrid.y = (size - 1) / BLOCK_SIZE + 1;
		elimination5<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 6:
		// GPU Kernel 6
		dimBlock.x = size + 1;
		dimBlock.y = size;
		elimination6<<<dimGrid, dimBlock>>>(g_a, g_b, size, 0);
		for (unsigned int i = 1; i < size; i++) {
			elimination6<<<dimGrid, dimBlock>>>(g_b, g_b, size, i);
		}
		break;
	case 7:
		// GPU Kernel 7
		dimBlock.x = size + 1;
		dimGrid.x = size;
		elimination7<<<dimGrid, dimBlock>>>(g_a, g_b, size, 0);
		for (unsigned int i = 1; i < size; i++) {
			elimination7<<<dimGrid, dimBlock>>>(g_b, g_b, size, i);
		}
		break;
	case 8:
		// GPU Kernel 8
		dimBlock.x = size + 1;
		dimBlock.y = size;
		for (unsigned int i = 0; i < size; i++) {
			elimination8_1<<<dimGrid, dimBlock>>>(g_b, size, i);
		}
		elimination8_2<<<dimGrid, dimBlock>>>(g_b, size);
		break;
	case 9:
		// GPU Kernel 9
		dimBlock.x = size + 1;
		dimBlock.y = size;
		elimination9<<<dimGrid, dimBlock>>>(g_b, size);
		break;
	case 10:
		// GPU Kernel 10
		dimBlock.x = size + 1;
		dimBlock.y = size;
		elimination10<<<dimGrid, dimBlock>>>(g_b, size);
		break;
	case 11:
		// GPU Kernel 11
		dimBlock.x = BLOCK_SIZE;
		dimBlock.y = BLOCK_SIZE;
		dimGrid.x = (size + 1 - 1) / BLOCK_SIZE + 1;
		dimGrid.y = (size - 1) / BLOCK_SIZE + 1;
		for (int pivot = 0; pivot < size; pivot++) {
			elimination11_1<<<dimGrid, dimBlock>>>(g_b, size, pivot);
		}
		dimBlock.y = 1;
		dimGrid.y = 1;
		elimination11_2<<<dimGrid, dimBlock>>>(g_b, size);
		break;
	case 12:
		// GPU Kernel 12
		dimBlock.x = 512;
		dimBlock.y = 1;
		dimGrid.x = (size + 1 - 1) / dimBlock.x + 1;
		dimGrid.y = (size - 1) / dimBlock.y + 1;
		for (int pivot = 0; pivot < size; pivot++) {
			elimination12_1<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			elimination12_2<<<dimGrid, dimBlock>>>(g_b, size, pivot);
		}
		break;
	case 13:
		// GPU Kernel 13
		dimBlock.x = 512;
		dimBlock.y = 1;
		dimGrid.x = (size + 1 - 1) / dimBlock.x + 1;
		dimGrid.y = (size - 1) / dimBlock.y + 1;
		for (int pivot = 0; pivot < size; pivot++) {
			elimination13_1<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			elimination13_2<<<dimGrid, dimBlock>>>(g_b, size, pivot);
		}
		break;
	case 14:
		// GPU Kernel 14
		dimBlock.x = 512;
		dimBlock.y = 1;
		dimGrid.x = (size + 1 - 1) / dimBlock.x + 1;
		dimGrid.y = (size - 1) / dimBlock.y + 1;
		for (int pivot = 0; pivot < size; pivot++) {
			elimination14_1<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			elimination14_2<<<dimGrid, dimBlock>>>(g_b, size, pivot);
		}
		break;
	case 15:
		// GPU Kernel 15
		dimBlock.x = 512;
		dimBlock.y = 1;
		dimGrid.x = (size + 1 - 1) / dimBlock.x + 1;
		dimGrid.y = (size - 1) / dimBlock.y + 1;
		for (int pivot = 0; pivot < size; pivot++) {
			elimination15_1<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			elimination15_2<<<dimGrid, dimBlock>>>(g_b, size, pivot);
		}
		break;
	case 16:
		// GPU Kernel 16
		dimBlock.x = 512;
		dimBlock.y = 1;
		dimGrid.x = (size + 1 - 1) / dimBlock.x + 1;
		dimGrid.y = (size - 1) / dimBlock.y + 1;
		for (int pivot = 0; pivot < size; pivot++) {
			elimination16_1<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			elimination16_2<<<dimGrid, dimBlock>>>(g_b, size, pivot);
		}
		break;
	}
	check("Executed kernel on GPU");

	cudaMemcpy(b, g_b, sizeTotal * sizeof(float), cudaMemcpyDeviceToHost);
	check("Copied data from device to host");

	// Tidy up
	check("Freeing memory");
	if (kernel < 8) {
		cudaFree(g_a);
		cudaFree(g_b);
	} else {
		cudaFree(g_b);
	}

	// Stop timers
	cudaEventRecord(timer2, 0);
	cudaEventSynchronize(timer1);
	cudaEventSynchronize(timer2);
	float elapsed;
	cudaEventElapsedTime(&elapsed, timer1, timer2);

	cudaDeviceReset();
	return elapsed;
}

// ----------------------------- elimination 0 ------------------------------ //
// This is a terrible solution which simply executes the CPU code upon the GPU.
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

// ----------------------------- elimination 1 ------------------------------ //
// Based upon elimination 0. Inner xx loops have been made parallel. Uses only
// one block, and uses global memory.
// Max size is 512
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

		__syncthreads();

		// Make all other values in the pivot column be zero
		for (rr = 0; rr < size; rr++) {
			if (rr != yy)
				element(xx, rr) -= element(yy, rr) * element(xx, yy);
		}

		__syncthreads();

	}
#undef element
}

// ----------------------------- elimination 2 ------------------------------ //
// Based upon elimination 1. Both xx and rr loops are now in parallel. Because
// the grid is now 2D, the max size has dropped from 511 to 22. This is because
// the max size is limited by the number of threads per block, which is 512.
// The number of threads required per size is ((size + 1) * size). 22 is the
// largest number for which this result is less than 512:
//   ((22 + 1) * 22) < 512, ((23 + 1) * 23) > 512; 23 will not fit, 22 will.
// Max size is 22
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

		__syncthreads();

		// Make all other values in the pivot column be zero
		if (rr != yy)
			element(xx, rr) -= element(yy, rr) * element(xx, yy);

		__syncthreads();
	}
#undef element
}

// ----------------------------- elimination 3 ------------------------------ //
// Based upon elimination 3. Data is copied in parallel.
// Max size is 22
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

// ----------------------------- elimination 4 ------------------------------ //
// Based upon elimination 3. Shared memory is used.
// Max size is 22
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

// ----------------------------- elimination 5 ------------------------------ //
// A new approach. Tries to use tiled implementation. Does not work.
// Max size is ???
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

// ----------------------------- elimination 6 ------------------------------ //
// Another new approach. Kernel is invoked once per pivot. There is just one
// block, with thread dimensions equal to matrix dimensions.
// Max size is 22
__global__ void elimination6(float *a, float *b, int size, int pivot) {
#define element(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))
	int x = threadIdx.x;
	int y = threadIdx.y;

	int tid = y * (size + 1) + x;
	b[tid] = a[tid];

	if (y == pivot)
		element(x, y) /= element(pivot, pivot);

	__syncthreads();

	if (y != pivot)
		element(x, y) -= element(pivot, y) * element(x, pivot);

#undef element
}

// ----------------------------- elimination 7 ------------------------------ //
// Based upon elimination 6. Uses multiple blocks, with fixed dimension sizes.
// Does not work.
// Max size is ???
__global__ void elimination7(float *a, float *b, int size, int pivot) {
#define element(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	int x = threadIdx.x;
	int y = blockDim.x;

	if (x < size + 1 && y < size) {
		int tid = y * (size + 1) + x;

		b[tid] = a[tid];

		__syncthreads();

		if (y == pivot)
			element(x, y) /= element(pivot, pivot);

		__syncthreads();

		if (y != pivot)
			element(x, y) -= element(pivot, y) * element(x, pivot);
	}

#undef element
}

// ----------------------------- elimination 8 ------------------------------ //
// Yet another new approach. Splits the problem into two kernels, and changes
// the logic of the algorithm slightly. The division and subtraction has been
// combined into one operation in part 8_1. However, the result must be divided
// one last time in part 8_2.
// Max size is 22
__global__ void elimination8_1(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
	int x = threadIdx.x;
	int y = threadIdx.y;

	float cp = element(pivot, y) / element(pivot, pivot);

	if (y != pivot)
		element(x, y) -= cp * element(x, pivot);

#undef element
}

__global__ void elimination8_2(float *a, int size) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
	int yy = threadIdx.y * (size + 1) + threadIdx.x;
	element(size, yy) /= element(yy, yy);
#undef element
}

// ----------------------------- elimination 9 ------------------------------ //
// Based upon elimination 8. Both parts 8_1 and 8_2 have been combined together.
// Max size is 22
__global__ void elimination9(float *a, int size) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
	int x = threadIdx.x;
	int y = threadIdx.y;

	float cp;

	for (int pivot = 0; pivot < size; pivot++) {

		cp = element(pivot, y) / element(pivot, pivot);

		if (y != pivot)
			element(x, y) -= cp * element(x, pivot);

		__syncthreads();
	}

	int yy = threadIdx.y * (size + 1) + threadIdx.x;
	element(size, yy) /= element(yy, yy);
#undef element
}

// ----------------------------- elimination 10 ----------------------------- //
// Based upon elimination 9. Uses shared memory. Limited by amount of shared
// memory per block.
// Max size is 22
__global__ void elimination10(float *a, int size) {
#define element(_x, _y) (*(sdata + ((_y) * (size + 1) + (_x))))

	int x = threadIdx.x;
	int y = threadIdx.y;
	int tid = y * (size + 1) + x;

	__shared__ float sdata[(22 + 1) * 22]; // Max size that will fit is 22
	sdata[tid] = a[tid];

	float cp;

	for (int pivot = 0; pivot < size; pivot++) {

		cp = element(pivot, y) / element(pivot, pivot);

		if (y != pivot)
			element(x, y) -= cp * element(x, pivot);

		__syncthreads();
	}

	element(size, tid) /= element(tid, tid);

	__syncthreads();

	a[tid] = sdata[tid];
#undef element
}

// ----------------------------- elimination 11 ----------------------------- //
// Loosely based upon elimination 8. Applies the same logic but uses a tiled
// implementation.
// Max size is ???
__global__ void elimination11_1(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > size || y > size - 1)
		return;

	float cp = element(pivot, y) / element(pivot, pivot);

	if (y != pivot)
		element(x, y) -= cp * element(x, pivot);

#undef element
}

__global__ void elimination11_2(float *a, int size) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid > size)
		return;

	element(size, tid) /= element(tid, tid);

#undef element
}

// ----------------------------- elimination 12 ----------------------------- //
// Based upon elimination 11. Each block contains 512x1 threads that operate on
// a row each.
// Max size is 511
__global__ void elimination12_1(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x <= size && y < size)
		if (y == pivot)
			element(x, y) /= element(pivot, pivot);

#undef element
}

__global__ void elimination12_2(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x <= size && y < size)
		if (y != pivot)
			element(x, y) -= element(pivot, y) * element(x, pivot);

#undef element
}

// ----------------------------- elimination 13 ----------------------------- //
// Based upon elimination 12. Conditions and calculations have been rearranged
// to ensure threads don't perform unnecessary work. Each thread calculates
// result for multiple elements.
// Max size is ((512 * ELEMENTS_PER_THREAD) - 1)
__global__ void elimination13_1(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int x = (threadIdx.x + blockIdx.x * blockDim.x) * ELEMENTS_PER_THREAD;
	int y = blockIdx.y;

	if (y < size)
		if (y == pivot) {
			float p = element(pivot, pivot);
			for (int xx = x; xx < x + ELEMENTS_PER_THREAD; xx++)
				if (xx <= size)
					element(xx, y) /= p;
		}

#undef element
}

__global__ void elimination13_2(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int x = (threadIdx.x + blockIdx.x * blockDim.x) * ELEMENTS_PER_THREAD;
	int y = blockIdx.y;

	if (y < size)
		if (y != pivot) {
			float c = element(pivot, y);
			for (int xx = x; xx < x + ELEMENTS_PER_THREAD; xx++)
				if (xx <= size)
					element(xx, y) -= c * element(xx, pivot);
		}

#undef element
}

// ----------------------------- elimination 14 ----------------------------- //
// Based upon elimination 13. Loops have been completely unrolled, and have
// been made slightly more efficient through caching 'xx'.
// Max size is ((512 * ELEMENTS_PER_THREAD) - 1)
__global__ void elimination14_1(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int y = blockIdx.y;

	if (y >= size || y != pivot)
		return;

	int x = (threadIdx.x + blockIdx.x * blockDim.x) * ELEMENTS_PER_THREAD;
	float p = element(pivot, pivot);
	int xx;

	#pragma unroll
	for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
		xx = i + x;
		if (xx <= size)
			element(xx, y) /= p;
	}

#undef element
}

__global__ void elimination14_2(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int y = blockIdx.y;

	if (y >= size || y == pivot)
		return;

	int x = (threadIdx.x + blockIdx.x * blockDim.x) * ELEMENTS_PER_THREAD;
	float c = element(pivot, y);
	int xx;

	#pragma unroll
	for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
		xx = i + x;
		if (xx <= size)
			element(xx, y) -= c * element(xx, pivot);
	}

#undef element
}

// ----------------------------- elimination 15 ----------------------------- //
// Based upon elimination 14. Access to matrix elements has been made more
// efficient through combining redundant operations.
// Max size is ((512 * ELEMENTS_PER_THREAD) - 1)
__global__ void elimination15_1(float *a, int size, int pivot) {

	int y = blockIdx.y;

	if (y >= size || y != pivot)
		return;

	int x = (threadIdx.x + blockIdx.x * blockDim.x) * ELEMENTS_PER_THREAD;

	//if (x < pivot)
	//	return;

	int w = size + 1;
	float p = *(a + pivot * w + pivot);

	if (x + ELEMENTS_PER_THREAD - 1 <= size) {
		// This is not an edge case. No bounds checking required.
		float *aywx = a + y * w + x;
		#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
			*(aywx + i) /= p;
	} else {
		// This is an edge case. Bounds checking is required.
		float *ayw = a + y * w;
		int xx;
		#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
			xx = i + x;
			if (xx <= size)
				*(ayw + xx) /= p;
		}
	}
}

__global__ void elimination15_2(float *a, int size, int pivot) {

	int y = blockIdx.y;

	if (y >= size || y == pivot)
		return;

	int x = (threadIdx.x + blockIdx.x * blockDim.x) * ELEMENTS_PER_THREAD;

	//if (x < pivot)
	//	return;

	int w = size + 1;
	int pivotw = pivot * w;
	float *ayw = a + y * w;
	float c = *(ayw + pivot);
	int xx;

	if (x + ELEMENTS_PER_THREAD - 1 <= size) {
		// This is not an edge case. No bounds checking is required.
		#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
			xx = i + x;
			*(ayw + xx) -= c * *(a + pivotw + xx);
		}
	} else {
		// This is an edge case. Bounds checking is required.
		#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
			xx = i + x;
			if (xx <= size)
				*(ayw + xx) -= c * *(a + pivotw + xx);
		}
	}
}

// ----------------------------- elimination 16 ----------------------------- //
// Based upon elimination 15. Removed ELEMENTS_PER_THREAD, which unfortunately
// decreases the max size back to 511, but opens up the possibility for several
// more optimizations. For example, we can now use the check (x < pivot) to drop
// some threads, because any x value left of the pivot will not affect the final
// outcome of the algorithm.
// Max size is 511
__global__ void elimination16_1(float *a, int size, int pivot) {

	int y = blockIdx.y;

	if (y >= size || y != pivot)
		return;

	int x = threadIdx.x;

	if (x < pivot)
		return;

	int w = size + 1;
	*(a + y * w + x) /= *(a + pivot * w + pivot);

}

__global__ void elimination16_2(float *a, int size, int pivot) {

	int y = blockIdx.y;

	if (y >= size || y == pivot)
		return;

	int x = threadIdx.x;

	if (x < pivot)
		return;

	int w = size + 1;
	int pivotw = pivot * w;
	float *ayw = a + y * w;

	float c = *(ayw + pivot);
	*(ayw + x) -= c * *(a + pivotw + x);

}
