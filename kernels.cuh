#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void predictModelKernel(int *data, double *predictions,double* residuals, double beta_0, double beta_1,double beta_2, int MAX_VARIABLES, int numObservations){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numObservations) {
        int offset = tid * MAX_VARIABLES;

        // Extract variables for the observation
        int y = data[offset];
        int x1 = data[offset + 1];
        int x2 = data[offset + 2];

        // Calculate the prediction using the provided beta coefficients
        predictions[tid] = beta_0 + beta_1 * x1 + beta_2 * x2;
        __syncthreads();
        residuals[tid]=y-predictions[tid];
    }
    
}

// CUDA kernel to copy array from device to device
__global__ void copyArrayKernel(int *data, int *data_copy, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for valid thread index
    if (tid < size) {
        data_copy[tid] = data[tid];
    }
}

__global__ void calculateSquareKernel(int* data, int* sum, int size,int xvar){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int index=3*tid+xvar;
    
    if (index<size){
        atomicAdd(sum,data[index]*data[index]);
    }
}


__global__ void calculateSumOfProductSquaredKernel(int* data, int* sum, int size,int var1,int var2){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int firstIndex=3*tid+var1;
    int secondIndex=3*tid+var2;

    if ((firstIndex<size)||(secondIndex<size)){
        atomicAdd(sum,data[firstIndex]*data[secondIndex]);
    }
}


__global__ void calculateVarSumKernel(int* data, int* sum, int size,int var){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int index=3*tid+var;

    if (index<size){
        // sum+=data[index]*data[index];
        atomicAdd(sum,data[index]);
    }
}