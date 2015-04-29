#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>

#pragma once

static void HandleError(cudaError_t err, const char* file, int line){
	if (err != cudaSuccess){
		printf("ERROR: %s in %s at line %d\n", cudaGetErrorString(err), file, line );
		exit(1);
	}
}

#define HANDLE_ERROR( err )  (HandleError(err, __FILE__, __LINE__)) 
