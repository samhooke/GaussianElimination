#include "elimination_gpu.h"

// Used by kernel 6
#define BLOCK_SIZE 16

// Used by kernels 8, 9 & 10
#define ELEMENTS_PER_THREAD 4

// Used by kernel 12
#define SHARED_SIZE 16

// Used by kernel 13, 14, 15, 16 & 17
// For kernels >= 14, BLOCK_WIDTH must be a factor of (size + 1)
// The optimal BLOCK_WIDTH was found to be 128
#define BLOCK_WIDTH 128

float elimination_gpu(float *a, float *b, int size, int kernel) {
	// Start timers
	check("Creating timers");
	cudaEvent_t timer1, timer2;
	cudaEventCreate(&timer1);
	cudaEventCreate(&timer2);
	cudaEventRecord(timer1, 0);
	cudaEventSynchronize(timer1);

	int sizeTotal = (size + 1) * size;
	float *g_a, *g_b;

	if (kernel < 5 || kernel > 11) {
		// Kernels 1 to 4 require two copies of the matrix
		cudaMalloc((void**)&g_a, sizeTotal * sizeof(float));
		check("Allocated memory g_a");
		cudaMalloc((void**)&g_b, sizeTotal * sizeof(float));
		check("Allocated memory g_b");

		// Copy a to g_a which the device will use for reference
		cudaMemcpy(g_a, a, sizeTotal * sizeof(float), cudaMemcpyHostToDevice);
		check("Copied memory from host to device");
	} else {
		// Kernels 5 to 11 require one copy of the matrix
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
	case 1:
		// GPU Kernel 1

		dimBlock.x = size + 1;

		gpu_kernel_1<<<dimGrid, dimBlock>>>(g_a, g_b, size);
		break;
	case 2:
	case 3:
	case 4:
		// GPU Kernel 2, 3 & 4

		dimBlock.x = size + 1;
		dimBlock.y = size;

		switch (kernel) {
		case 2:
			gpu_kernel_2<<<dimGrid, dimBlock>>>(g_a, g_b, size);
			break;
		case 3:
			gpu_kernel_3<<<dimGrid, dimBlock>>>(g_a, g_b, size);
			break;
		case 4:
			gpu_kernel_4<<<dimGrid, dimBlock>>>(g_a, g_b, size);
			break;
		}
		break;
	case 5:
		// GPU Kernel 5

		dimBlock.x = size + 1;
		dimBlock.y = size;

		for (unsigned int i = 0; i < size; i++) {
			gpu_kernel_5a<<<dimGrid, dimBlock>>>(g_b, size, i);
		}
		gpu_kernel_5b<<<dimGrid, dimBlock>>>(g_b, size);
		break;
	case 6:
		// GPU Kernel 6

		dimBlock.x = BLOCK_SIZE;
		dimBlock.y = BLOCK_SIZE;
		dimGrid.x = (size + 1 - 1) / BLOCK_SIZE + 1;
		dimGrid.y = (size - 1) / BLOCK_SIZE + 1;

		for (int pivot = 0; pivot < size; pivot++) {
			gpu_kernel_6a<<<dimGrid, dimBlock>>>(g_b, size, pivot);
		}

		dimBlock.y = 1;
		dimGrid.y = 1;

		gpu_kernel_6b<<<dimGrid, dimBlock>>>(g_b, size);
		break;
	case 7:
	case 8:
	case 9:
	case 10:
	case 11:
		// GPU Kernel 7, 8, 9, 10 & 11

		dimBlock.x = 512;
		dimBlock.y = 1;
		dimGrid.x = (size + 1 - 1) / dimBlock.x + 1;
		dimGrid.y = (size - 1) / dimBlock.y + 1;

		switch (kernel) {
		case 7:
			for (int pivot = 0; pivot < size; pivot++) {
				gpu_kernel_7a<<<dimGrid, dimBlock>>>(g_b, size, pivot);
				gpu_kernel_7b<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			}
			break;
		case 8:
			for (int pivot = 0; pivot < size; pivot++) {
				gpu_kernel_8a<<<dimGrid, dimBlock>>>(g_b, size, pivot);
				gpu_kernel_8b<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			}
			break;
		case 9:
			for (int pivot = 0; pivot < size; pivot++) {
				gpu_kernel_9a<<<dimGrid, dimBlock>>>(g_b, size, pivot);
				gpu_kernel_9b<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			}
			break;
		case 10:
			for (int pivot = 0; pivot < size; pivot++) {
				gpu_kernel_10a<<<dimGrid, dimBlock>>>(g_b, size, pivot);
				gpu_kernel_10b<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			}
			break;
		case 11:
			for (int pivot = 0; pivot < size; pivot++) {
				gpu_kernel_11a<<<dimGrid, dimBlock>>>(g_b, size, pivot);
				gpu_kernel_11b<<<dimGrid, dimBlock>>>(g_b, size, pivot);
			}
			break;
		}
		break;
	case 12:
		// GPU Kernel 12

		dimBlock.x = SHARED_SIZE;
		dimBlock.y = SHARED_SIZE;
		dimGrid.x = (size + 1 - 1) / dimBlock.x + 1;
		dimGrid.y = (size - 1) / dimBlock.y + 1;

		for (int pivot = 0; pivot < size; pivot++) {
			gpu_kernel_12a<<<dimGrid, dimBlock>>>(g_a, g_b, size, pivot);
			gpu_kernel_12b<<<dimGrid, dimBlock>>>(g_b, g_a, size, pivot);
		}
		break;
	case 13:
	case 14:
	case 15:
		// GPU Kernel 13, 14 & 15

		dimBlock.x = BLOCK_WIDTH;
		dimBlock.y = 1;
		dimGrid.x = (size + 1 - 1) / BLOCK_WIDTH + 1;
		dimGrid.y = size;

		switch (kernel) {
		case 13:
			for (int pivot = 0; pivot < size; pivot++) {
				gpu_kernel_13a<<<dimGrid, dimBlock>>>(g_a, g_b, size, pivot);
				gpu_kernel_13b<<<dimGrid, dimBlock>>>(g_b, g_a, size, pivot);
			}
			break;
		case 14:
			for (int pivot = 0; pivot < size; pivot++) {
				gpu_kernel_14a<<<dimGrid, dimBlock>>>(g_a, g_b, size, pivot);
				gpu_kernel_14b<<<dimGrid, dimBlock>>>(g_b, g_a, size, pivot);
			}
			break;
		case 15:
			for (int pivot = 0; pivot < size; pivot++) {
				gpu_kernel_15a<<<dimGrid, dimBlock>>>(g_a, g_b, size, pivot);
				gpu_kernel_15b<<<dimGrid, dimBlock>>>(g_b, g_a, size, pivot);
			}
			break;
		}
		break;
	case 16:
	case 17:
		// GPU Kernel 16 & 17

		int gx = size / BLOCK_WIDTH + 1;
		dimBlock.x = BLOCK_WIDTH;
		dimBlock.y = 1;
		dimGrid.x = gx;
		dimGrid.y = size;

		int xoffset = 0;

		switch (kernel) {
		case 16:
			for (int pivot = 0; pivot < size; pivot++) {
				xoffset = floor(pivot / BLOCK_WIDTH);
				dimGrid.x = gx - xoffset;
				xoffset *= BLOCK_WIDTH;
				gpu_kernel_16a<<<dimGrid, dimBlock>>>(g_a, g_b, size, pivot, xoffset);
				gpu_kernel_16b<<<dimGrid, dimBlock>>>(g_b, g_a, size, pivot, xoffset);
			}
			break;
		case 17:
			int s = size + 1;
			for (int pivot = 0; pivot < size; pivot++) {
				xoffset = floor(pivot / BLOCK_WIDTH);
				dimGrid.x = gx - xoffset;
				xoffset *= BLOCK_WIDTH;
				gpu_kernel_17a<<<dimGrid, dimBlock>>>(g_a, g_b, s, pivot, xoffset);
				gpu_kernel_17b<<<dimGrid, dimBlock>>>(g_b, g_a, s, pivot, xoffset);
			}
			break;
		}
		break;
	}
	cudaDeviceSynchronize();
	check("Executed kernel on GPU");

	if (kernel >= 12)
		cudaMemcpy(b, g_a, sizeTotal * sizeof(float), cudaMemcpyDeviceToHost);
	else
		cudaMemcpy(b, g_b, sizeTotal * sizeof(float), cudaMemcpyDeviceToHost);
	check("Copied data from device to host");

	// Tidy up
	check("Freeing memory");
	if (kernel < 5) {
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

// ----------------------------- GPU Kernel 1 ------------------------------- //
// Based upon CPU Kernel 1. Inner xx loops have been made parallel. Uses
// only one block, and uses global memory.
// Max size is 511
__global__ void gpu_kernel_1(float *a, float *b, int size) {
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

// ----------------------------- GPU Kernel 2 ------------------------------- //
// Based upon GPU Kernel 1. Both xx and rr loops are now in parallel. Because
// the grid is now 2D, the max size has dropped from 511 to 22. This is because
// the max size is limited by the number of threads per block, which is 512.
// The number of threads required per size is ((size + 1) * size). 22 is the
// largest number for which this result is less than 512:
//   ((22 + 1) * 22) < 512, ((23 + 1) * 23) > 512; 23 will not fit, 22 will.
// Max size is 22
__global__ void gpu_kernel_2(float *a, float *b, int size) {
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

// ----------------------------- GPU Kernel 3 ------------------------------- //
// Based upon GPU Kernel 2. Data is copied in parallel.
// Max size is 22
__global__ void gpu_kernel_3(float *a, float *b, int size) {
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

// ----------------------------- GPU Kernel 4 ------------------------------- //
// Based upon GPU Kernel 3. Shared memory is used.
// Max size is 22
__global__ void gpu_kernel_4(float *a, float *b, int size) {
#define element(_x, _y) (*(sdata + ((_y) * (size + 1) + (_x))))
	unsigned int xx, yy, rr;

	// With a limit of 512 threads per block, and only one block, this results
	//in a maximum of a matrix size 22, which requires (22 + 1) x 22 values
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

// ----------------------------- GPU Kernel 5 ------------------------------- //
// Yet another new approach. Splits the problem into two kernels, and changes
// the logic of the algorithm slightly. The division and subtraction has been
// combined into one operation in part 5a. However, the result must be divided
// one last time in part 5b.
// Max size is 22
__global__ void gpu_kernel_5a(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
	int x = threadIdx.x;
	int y = threadIdx.y;

	float cp = element(pivot, y) / element(pivot, pivot);

	if (y != pivot)
		element(x, y) -= cp * element(x, pivot);

#undef element
}

__global__ void gpu_kernel_5b(float *a, int size) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
	int yy = threadIdx.y * (size + 1) + threadIdx.x;
	element(size, yy) /= element(yy, yy);
#undef element
}

// ----------------------------- GPU Kernel 6 ------------------------------- //
// Loosely based upon GPU Kernel 5. Applies the same logic but uses a tiled
// implementation.
// Max size is ???
__global__ void gpu_kernel_6a(float *a, int size, int pivot) {
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

__global__ void gpu_kernel_6b(float *a, int size) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid > size)
		return;

	element(size, tid) /= element(tid, tid);

#undef element
}

// ----------------------------- GPU Kernel 7 ------------------------------- //
// Based upon GPU Kernel 6. Each block contains 512x1 threads that operate on
// a row each.
// Max size is 511
__global__ void gpu_kernel_7a(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x <= size && y < size)
		if (y == pivot)
			element(x, y) /= element(pivot, pivot);

#undef element
}

__global__ void gpu_kernel_7b(float *a, int size, int pivot) {
#define element(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x <= size && y < size)
		if (y != pivot)
			element(x, y) -= element(pivot, y) * element(x, pivot);

#undef element
}

// ----------------------------- GPU Kernel 8 ------------------------------- //
// Based upon GPU Kernel 7. Conditions and calculations have been rearranged
// to ensure threads don't perform unnecessary work. Each thread calculates
// result for multiple elements.
// Max size is ((512 * ELEMENTS_PER_THREAD) - 1)
__global__ void gpu_kernel_8a(float *a, int size, int pivot) {
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

__global__ void gpu_kernel_8b(float *a, int size, int pivot) {
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

// ----------------------------- GPU Kernel 9 ------------------------------- //
// Based upon GPU Kernel 8. Loops have been completely unrolled, and have
// been made slightly more efficient through caching 'xx'.
// Max size is ((512 * ELEMENTS_PER_THREAD) - 1)
__global__ void gpu_kernel_9a(float *a, int size, int pivot) {
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

__global__ void gpu_kernel_9b(float *a, int size, int pivot) {
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

// ----------------------------- GPU Kernel 10 ------------------------------ //
// Based upon GPU Kernel 9. Access to matrix elements has been made more
// efficient through combining redundant operations.
// Max size is ((512 * ELEMENTS_PER_THREAD) - 1)
__global__ void gpu_kernel_10a(float *a, int size, int pivot) {

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

__global__ void gpu_kernel_10b(float *a, int size, int pivot) {

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

// ----------------------------- GPU Kernel 11 ------------------------------ //
// Based upon GPU Kernel 10. Removed ELEMENTS_PER_THREAD, which unfortunately
// decreases the max size back to 511, but opens up the possibility for several
// more optimizations. For example, we can now use the check (x < pivot) to drop
// some threads, because any x value left of the pivot will not affect the final
// outcome of the algorithm.
// Max size is 511
__global__ void gpu_kernel_11a(float *a, int size, int pivot) {

	int y = blockIdx.y;

	if (y >= size || y != pivot)
		return;

	int x = threadIdx.x;

	if (x < pivot)
		return;

	int w = size + 1;
	*(a + y * w + x) /= *(a + pivot * w + pivot);

}

__global__ void gpu_kernel_11b(float *a, int size, int pivot) {

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

// ----------------------------- GPU Kernel 12 ------------------------------ //
// A complete rewrite using a tiled implementation and shared memory.
__global__ void gpu_kernel_12a(float *a, float *b, int size, int pivot) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float p;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int x = tx + bx * SHARED_SIZE;
	int y = ty + by * SHARED_SIZE;

	if (x >= size + 1 || y >= size)
		return;

	if (tx == 0)
		p = mread(pivot, pivot);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y) / p;
	else
		mwrite(x, y) = mread(x, y);

#undef mread
#undef mwrite
}

__global__ void gpu_kernel_12b(float *a, float *b, int size, int pivot) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float row[SHARED_SIZE];
	__shared__ float col[SHARED_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int x = tx + bx * blockDim.x;
	int y = ty + by * blockDim.y;

	if (x >= size + 1 || y >= size)
		return;

	__syncthreads();

	row[tx] = mread(x, pivot);
	col[ty] = mread(pivot, y);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y);
	else
		mwrite(x, y) = mread(x, y) - col[ty] * row[tx];

#undef mread
#undef mwrite
}

// ----------------------------- GPU Kernel 13 ------------------------------ //
// Based upon GPU Kernel 12. Works per row instead of per tile. This avoids
// any divergence occurring.
__global__ void gpu_kernel_13a(float *a, float *b, int size, int pivot) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float p;

	int x = threadIdx.x + blockIdx.x * BLOCK_WIDTH;
	int y = blockIdx.y;

	if (x >= size + 1)
		return;

	if (threadIdx.x == 0)
		p = mread(pivot, pivot);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y) / p;
	else
		mwrite(x, y) = mread(x, y);

#undef mread
#undef mwrite
}

__global__ void gpu_kernel_13b(float *a, float *b, int size, int pivot) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float row[BLOCK_WIDTH];
	__shared__ float col;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = blockIdx.y;

	if (x >= size + 1)
		return;

	row[threadIdx.x] = mread(x, pivot);

	if (threadIdx.x == 0)
		col = mread(pivot, y);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y);
	else
		mwrite(x, y) = mread(x, y) - col * row[threadIdx.x];

#undef mread
#undef mwrite
}

// ----------------------------- GPU Kernel 14 ------------------------------ //
// Based upon GPU Kernel 13. Does not check for x bounds, so BLOCK_WIDTH must
// be a factor of (matrix size + 1) e.g. size = 1023, BLOCK_WIDTH = 256. Also
// no longer uses shared memory for storing the row, as this is only read once.
__global__ void gpu_kernel_14a(float *a, float *b, int size, int pivot) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float p;

	int x = threadIdx.x + blockIdx.x * BLOCK_WIDTH;
	int y = blockIdx.y;

	if (threadIdx.x == 0)
		p = mread(pivot, pivot);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y) / p;
	else
		mwrite(x, y) = mread(x, y);

#undef mread
#undef mwrite
}

__global__ void gpu_kernel_14b(float *a, float *b, int size, int pivot) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float col;

	int x = threadIdx.x + blockIdx.x * BLOCK_WIDTH;
	int y = blockIdx.y;

	if (threadIdx.x == 0)
		col = mread(pivot, y);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y);
	else
		mwrite(x, y) = mread(x, y) - col * mread(x, pivot);

#undef mread
#undef mwrite
}

// ----------------------------- GPU Kernel 15 ------------------------------ //
// Based upon GPU Kernel 14. Avoids calculating elements for tiles that are
// left of the pivot, because this does not affect the final column. This is
// done on a per-tile bases to avoid divergence.
__global__ void gpu_kernel_15a(float *a, float *b, int size, int pivot) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float p;

	int bxw = blockIdx.x * BLOCK_WIDTH;

	if (bxw + BLOCK_WIDTH < pivot)
		return;

	int x = threadIdx.x + bxw;
	int y = blockIdx.y;

	if (threadIdx.x == 0)
		p = mread(pivot, pivot);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y) / p;
	else
		mwrite(x, y) = mread(x, y);

#undef mread
#undef mwrite
}

__global__ void gpu_kernel_15b(float *a, float *b, int size, int pivot) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float col;

	int bxw = blockIdx.x * BLOCK_WIDTH;

	if (bxw + BLOCK_WIDTH < pivot)
		return;

	int x = threadIdx.x + bxw;
	int y = blockIdx.y;

	if (threadIdx.x == 0)
		col = mread(pivot, y);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y);
	else
		mwrite(x, y) = mread(x, y) - col * mread(x, pivot);

#undef mread
#undef mwrite
}

// ----------------------------- GPU Kernel 16 ------------------------------ //
// Based upon GPU Kernel 15. Applies the same idea of avoiding calculating for
// x values < pivot, but achieves this through launching less threads rather
// than dropping unnecessary threads after launch. This is controlled through
// the xoffset parameter.
__global__ void gpu_kernel_16a(float *a, float *b, int size, int pivot, int xoffset) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float p;

	int x = threadIdx.x + blockIdx.x * BLOCK_WIDTH + xoffset;
	int y = blockIdx.y;

	if (threadIdx.x == 0)
		p = mread(pivot, pivot);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y) / p;
	else
		mwrite(x, y) = mread(x, y);

#undef mread
#undef mwrite
}

__global__ void gpu_kernel_16b(float *a, float *b, int size, int pivot, int xoffset) {
#define mread(_x, _y) (*(a + ((_y) * (size + 1) + (_x))))
#define mwrite(_x, _y) (*(b + ((_y) * (size + 1) + (_x))))

	__shared__ float col;

	int x = threadIdx.x + blockIdx.x * BLOCK_WIDTH + xoffset;
	int y = blockIdx.y;

	if (threadIdx.x == 0)
		col = mread(pivot, y);

	__syncthreads();

	if (y == pivot)
		mwrite(x, y) = mread(x, y);
	else
		mwrite(x, y) = mread(x, y) - col * mread(x, pivot);

#undef mread
#undef mwrite
}

// ----------------------------- GPU Kernel 17 ------------------------------ //
// Based upon GPU Kernel 16. Access to matrix data is more efficient. Also, the
// value of size is incremented by 1 before being passed through the function
// arguments, because (size + 1) was always used in the GPU code.
__global__ void gpu_kernel_17a(float *a, float *b, int size, int pivot, int xoffset) {

	__shared__ float p;

	int x = threadIdx.x + blockIdx.x * BLOCK_WIDTH + xoffset;
	int ysx = blockIdx.y * size + x;

	if (threadIdx.x == 0)
		p = *(a + pivot * (size + 1)); // pivot * size + pivot = pivot * (size + 1)

	__syncthreads();

	if (blockIdx.y == pivot)
		*(b + ysx) = *(a + ysx) / p;
	else
		*(b + ysx) = *(a + ysx);
}

__global__ void gpu_kernel_17b(float *a, float *b, int size, int pivot, int xoffset) {

	__shared__ float col;

	int x = threadIdx.x + blockIdx.x * BLOCK_WIDTH + xoffset;
	int ysx = blockIdx.y * size + x;

	if (threadIdx.x == 0)
		col = *(a + blockIdx.y * size + pivot);

	__syncthreads();

	if (blockIdx.y == pivot)
		*(b + ysx) = *(a + ysx);
	else
		*(b + ysx) = *(a + ysx) - col * *(a + pivot * size + x);
}

