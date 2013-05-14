#include "check.h"

#define SHOW_ALL_CHECKS true

void check(char* msg) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		fprintf(stderr, "Error: (%s) %s\n", msg, cudaGetErrorString(err));
	else if (SHOW_ALL_CHECKS)
		fprintf(stdout, "Log: (%s) %s\n", msg, cudaGetErrorString(err));
}
