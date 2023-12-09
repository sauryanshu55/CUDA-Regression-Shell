#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_VARIABLES 3
#define MAX_DATA_POINTS 1000
#define MAX_VARIABLE_NAME_LENGTH 50
#define MAX_COMMAND_LENGTH 100

// Global array to store data
int data[MAX_VARIABLES * MAX_DATA_POINTS];

int data_cpy[MAX_VARIABLES * MAX_DATA_POINTS];

// Array to store variable names
char variableNames[MAX_VARIABLES][256];

int numVars;
int numObservations;
bool data_primed = false;


typedef struct calculationInfo_t{
    int sumSquaredX1;
    int sumSquaredX2;
    int sumX1X2;
    int sumX1Y;
    int sumX2Y;
} calculationInfo_t;

calculationInfo_t globalCalculationInfo;

// Function to print the imported data
void printCSVData(int* data_to_copy) {
    printf("Variable Names:\n");
    for (int i = 0; i < numVars; i++) {
        printf("%s\t", variableNames[i]);
    }

    printf("\n");

    for (int j = 0; j < numObservations+1; j++) {
        for (int i = 0; i < numVars; i++) {
            printf("%d\t", data_to_copy[j * numVars + i]);
        }
        printf("\n");
    }
}

// Function to read CSV file and initialize data array
int readCSV(const char *filename) {
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        return -1;
    }

    // Read the first row to get variable names
    char line[1024];
    if (fgets(line, sizeof(line), file) != NULL) {
        // Tokenize the line to get variable names
        char *token = strtok(line, ",");
        int variableIndex = 0;

        while (token != NULL && variableIndex < MAX_VARIABLES) {
            // Remove leading and trailing whitespaces
            sscanf(token, " %[^ \t\n]", variableNames[variableIndex]);

            // Print variable names if needed
            // printf("Variable %d: %s\n", variableIndex + 1, variableNames[variableIndex]);

            token = strtok(NULL, ",");
            variableIndex++;
        }
        numVars = variableIndex;
    }

    // Read the rest of the file to get numerical data
    int dataPointIndex = 0;
    while (fgets(line, sizeof(line), file) != NULL && dataPointIndex < MAX_DATA_POINTS) {
        char *token = strtok(line, ",");
        int variableIndex = 0;

        while (token != NULL && variableIndex < MAX_VARIABLES) {
            sscanf(token, " %d", &data[dataPointIndex * numVars + variableIndex]);

            // Print data if needed
            // printf("%s: %d\n", variableNames[variableIndex], data[dataPointIndex * numVars + variableIndex]);

            token = strtok(NULL, ",");
            variableIndex++;
        }

        dataPointIndex++;
    }
    numObservations = dataPointIndex - 1;

    fclose(file);
    return 0;
}

// CUDA kernel to copy array from device to device
__global__ void copyArrayKernel(int *data, int *data_copy, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for valid thread index
    if (tid < size) {
        data_copy[tid] = data[tid];
    }
}


int copyDataToGPU(){
    int *gpu_data, *gpu_data_cpy;
    
    cudaMalloc((void**)&gpu_data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));
    cudaMalloc((void**)&gpu_data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));

    cudaMemcpy(gpu_data,data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    copyArrayKernel<<<gridSize,blockSize>>>(gpu_data,gpu_data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS));
    
    cudaDeviceSynchronize();

    cudaMemcpy(data_cpy,gpu_data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyDeviceToHost);
    
    // printCSVData(data_cpy);
    cudaFree(gpu_data);
    cudaFree(gpu_data_cpy);
    return 1;
}

__global__ void calculateSquareKernel(int* data, int* sum, int size,int xvar){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int index=3*tid+xvar;
    
    if (index<size){
        // sum+=data[index]*data[index];
        atomicAdd(sum,data[index]*data[index]);
    }
}

int calculateVarSquared(int xvar){
    int *gpu_data;
    int *gpu_result;
    int sum=0;

    cudaMalloc((void**)&gpu_data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));
    cudaMemcpy(gpu_data,data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_result,sizeof(int));
    cudaMemcpy(gpu_result,&sum,sizeof(int),cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    calculateSquareKernel<<<gridSize,blockSize>>>(gpu_data,gpu_result,(MAX_VARIABLES*MAX_DATA_POINTS),xvar);

    cudaDeviceSynchronize();
    cudaMemcpy(&sum,gpu_result,sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(gpu_data);
    cudaFree(gpu_result);

    if(xvar==1)  globalCalculationInfo.sumSquaredX1=sum;
    if(xvar==2) globalCalculationInfo.sumSquaredX2=sum;

    return 0;
}

__global__ void calculateSumOfProductSquaredKernel(int* data, int* sum, int size,int var1,int var2){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int firstIndex=3*tid+var1;
    int secondIndex=3*tid+var2;

    if ((firstIndex<size)||(secondIndex<size)){
        atomicAdd(sum,data[firstIndex]*data[secondIndex]);
    }
}


int calculateSumOfProductSquared(int var1, int var2){
    int *gpuData,*gpuResult;
    int sum=0;

    cudaMalloc((void**)&gpuData,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));
    cudaMemcpy(gpuData,data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpuResult,sizeof(int));
    cudaMemcpy(gpuResult,&sum,sizeof(int),cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    calculateSumOfProductSquaredKernel<<<gridSize,blockSize>>>(gpuData,gpuResult,(MAX_VARIABLES*MAX_DATA_POINTS),var1,var2);
    
    cudaDeviceSynchronize();

    cudaMemcpy(&sum,gpuResult,sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(gpuResult);
    cudaFree(gpuData);

    if ((var1==0)&&(var2==1)) globalCalculationInfo.sumX1Y=sum;
    if ((var1==0)&&(var2==2)) globalCalculationInfo.sumX2Y=sum;
    if ((var1==1)&&(var2==2)) globalCalculationInfo.sumX1X2=sum;

    return 0;
}

int runRegression(){
    calculateVarSquared(1);
    calculateVarSquared(2);
    calculateSumOfProductSquared(0,1);
    calculateSumOfProductSquared(0,2);
    calculateSumOfProductSquared(1,2);

    return 1;
}


int executeCommand(char command[]) {
    // exit
    if (strcmp(command, "exit") == 0) {
        printf("Exiting program\n");
        return -1;
    }

    // Check if the first part of the command is "load"
    if (strncmp(command, "load", 4) == 0) {
        char file_loc[100];
        // Extract the rest of the command string after "load"
        sscanf(command, "%*s %s", file_loc);

        // successful read=0
        if (readCSV(file_loc) == 0) {
            printf("Read CSV file from: %s \n", file_loc);
            data_primed = true;
        } else {
            printf("No such CSV file exists AND/OR Error in reading CSV File\n");
        }
        return 1;
    }

    // view
    if (strcmp(command, "view") == 0) {
        if (data_primed) {
            printCSVData(data);
            printf("Number of vars: %d\nNumber of observations: %d\n",numVars,numObservations);
        } else {
            printf("Data is not loaded yet\n");
        }
        return 1;
    }


    if (strcmp(command,"def")==0){
        readCSV("csv.csv");
        data_primed=true;
        printf("Read CSV file from: csv.csv \n");
        copyDataToGPU();
        runRegression();
        return 1;
    }    
    // Unrecognized command
    return 0;
}

int main() {
    char command[MAX_COMMAND_LENGTH];

    while (1) {
        // Print a prompt
        printf("$$> ");

        // Read a command from the user
        if (fgets(command, MAX_COMMAND_LENGTH, stdin) == NULL) {
            perror("Error reading command");
            exit(EXIT_FAILURE);
        }

        // Remove the newline character from the end of the command
        size_t length = strlen(command);
        if (length > 0 && command[length - 1] == '\n') {
            command[length - 1] = '\0';
        }

        // Execute the command using the system function
        int result = executeCommand(command);

        // Check if the command execution was successful
        if (result == 0) {
            printf("Unrecognized command\n");
        }

        // exit
        if (result == -1) {
            return 0;
        }
    }
    return 0;
}

