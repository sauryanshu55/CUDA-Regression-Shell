/**
kernels.cuh
--------------------------------------------------------------------------
DESCRPTION:
All the Nvidia GPU code has been defined in this file.
The specific kernels of the wrappers called in regression.cu call this CUDA header file.
--------------------------------------------------------------------------
*/
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cmath>

/*
GPU Kernel to predict model, given beta coefficients
*/
__global__ void predictModelKernel(int *data, int *predictions,int* residuals, double beta_0, double beta_1,double beta_2, int MAX_VARIABLES, int numObservations){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numObservations) {
        int offset = tid * MAX_VARIABLES; // Calculate Offset so that we can extract relevant variables

        // Extract variables for the observation
        int y = data[offset];
        int x1 = data[offset + 1];
        int x2 = data[offset + 2];

        // Calculate the prediction using the provided beta coefficients
        predictions[tid] = beta_0 + beta_1 * x1 + beta_2 * x2;

        // Sync Threads so that race conditions does not happen
        __syncthreads();

        // Calculate Residuals
        residuals[tid]=y-predictions[tid];
    }
    
}

/*
Calculate Standard Errors on GPU
with provided residuals, residual sum, variance array, residual variaance arrays
and degrees of freedom
*/
__global__ void calculateStandardErrorsKernel(int *residuals, int *residualSum, int *varianceArr,double *residualVariance, int degreesOfFreedom, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid<size){
        // Atomic Add
        atomicAdd(residualSum,residuals[tid]);

        // Sync threads so that the output from atomic add can be used
        __syncthreads();

        double mean=(double)residualSum[0]/(double)degreesOfFreedom;

        // Atomic Add
        atomicAdd(varianceArr,(int)pow((double)residuals[tid]-mean,2));

        // Sync threads so that the output from atomic add can be used
        __syncthreads();

        // Calculate Residual variance
         *residualVariance= std::fma((double)varianceArr[0],1.0/(degreesOfFreedom-1),0.0);
    }

}

/**
Calculate Variance of each of the given variables
*/
__global__ void calculateVarVarianceKernel(int *data, int *numeratorSumArr, double *varVariance,int degreesOfFreedom, int var, int mean, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Get variable indexes in the 1D array
    int index=3*tid+var;
    if (index<size){
        // Atomic Add
        if (data[index] != 0) atomicAdd(numeratorSumArr,(int)pow((double)data[index]-(double)mean,2));

    }

    // Sync threads so that variamce can be calculated. Prevent race condtions
    __syncthreads();

    *varVariance= std::fma((double)numeratorSumArr[0],1.0/(degreesOfFreedom-1),0.0);
    
    __syncthreads();
}


// CUDA kernel to copy array from device to device
__global__ void copyArrayKernel(int *data, int *data_copy, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for valid thread index
    if (tid < size) {
        data_copy[tid] = data[tid];
    }
}

// Cuda kernel to calculate square of a given variable
__global__ void calculateSquareKernel(int* data, int* sum, int size,int xvar){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Get variable indexes in the 1D array
    int index=3*tid+xvar;
    
    if (index<size){
        atomicAdd(sum,data[index]*data[index]);
    }
}

// Cuda kernel to calculate sum of square of a given variable
__global__ void calculateSumOfProductSquaredKernel(int* data, int* sum, int size,int var1,int var2){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Get variable indexes in the 1D array
    int firstIndex=3*tid+var1;
    int secondIndex=3*tid+var2;

    if ((firstIndex<size)||(secondIndex<size)){
        atomicAdd(sum,data[firstIndex]*data[secondIndex]);
    }
}

// Cuda kernel to calculate sum of variance square of a given variable
__global__ void calculateVarSumKernel(int* data, int* sum, int size,int var){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Get variable indexes in the 1D array
    int index=3*tid+var;

    if (index<size){
        // sum+=data[index]*data[index];
        atomicAdd(sum,data[index]);
    }
}

// CUDA kernel to index data as given per user regression instructions
__global__ void createdIndexedDataKernel(int *data, int *data_copy,int indexToPut, int actualIndex, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Get variable indexes in the 1D array
    int index=3*tid+indexToPut;
    int actual=3*tid+actualIndex;
    // Check for valid thread index
    if (index < size-3) {
        data_copy[index] = data[actual];
    }
}