#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// CUDA kernel to copy array from device to device
__global__ void copyArrayKernel(int *data, int *data_copy, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for valid thread index
    if (tid < size) {
        data_copy[tid] = data[tid];
    }
    __syncthreads();
}

__global__ void calculateSquareKernel(int* data, int* sum, int size,int xvar){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int index=3*tid+xvar;
    
    if (index<size){
        atomicAdd(sum,data[index]*data[index]);
    }
    __syncthreads();
}


__global__ void calculateSumOfProductSquaredKernel(int* data, int* sum, int size,int var1,int var2){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int firstIndex=3*tid+var1;
    int secondIndex=3*tid+var2;

    if ((firstIndex<size)||(secondIndex<size)){
        atomicAdd(sum,data[firstIndex]*data[secondIndex]);
    }
    __syncthreads();
}


__global__ void calculateVarSumKernel(int* data, int* sum, int size,int var){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int index=3*tid+var;

    if (index<size){
        // sum+=data[index]*data[index];
        atomicAdd(sum,data[index]);
    }
    __syncthreads();
}