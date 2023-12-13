#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "regression_info.cuh"

#define MAX_VARIABLES 3
#define MAX_DATA_POINTS 1000
#define MAX_VARIABLE_NAME_LENGTH 50
#define MAX_COMMAND_LENGTH 100

data_t Data;
bool data_primed = false;

calculationInfo_t globalCalculationInfo;
regressionInfo_t globalRegressionInfo;

// Function to print the imported data
void printCSVData(int* data) {
    printf("Variable Names:\n");
    for (int i = 0; i < Data.numVars; i++) {
        printf("%s\t", Data.variableNames[i]);
    }

    printf("\n");

    for (int j = 0; j < Data.numObservations; j++) {
        for (int i = 0; i < Data.numVars; i++) {
            printf("%d\t", data[j * Data.numVars + i]);
        }
        printf("\n");
    }
}

// Function to print the imported data
void printCSVData_d(double* data) {
    printf("Variable Names:\n");
    for (int i = 0; i < Data.numVars; i++) {
        printf("%s\t", Data.variableNames[i]);
    }

    printf("\n");

    for (int j = 0; j < Data.numObservations; j++) {
        for (int i = 0; i < Data.numVars; i++) {
            printf("%lf\t", data[j * Data.numVars + i]);
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
            sscanf(token, " %[^ \t\n]", Data.variableNames[variableIndex]);

            // Print variable names if needed
            // printf("Variable %d: %s\n", variableIndex + 1, Data.variableNames[variableIndex]);

            token = strtok(NULL, ",");
            variableIndex++;
        }
        Data.numVars = variableIndex;
    }

    // Read the rest of the file to get numerical data
    int dataPointIndex = 0;
    while (fgets(line, sizeof(line), file) != NULL && dataPointIndex < MAX_DATA_POINTS) {
        char *token = strtok(line, ",");
        int variableIndex = 0;

        while (token != NULL && variableIndex < MAX_VARIABLES) {
            sscanf(token, " %d", &Data.data[dataPointIndex * Data.numVars + variableIndex]);

            // Print data if needed
            // printf("%s: %d\n", Data.variableNames[variableIndex], data[dataPointIndex * Data.numVars + variableIndex]);

            token = strtok(NULL, ",");
            variableIndex++;
        }

        dataPointIndex++;
    }
    Data.numObservations = dataPointIndex;

    fclose(file);
    return 0;
}


int copyDataToGPU(){
    int *gpu_data, *gpu_data_cpy;
    
    cudaMalloc((void**)&gpu_data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));
    cudaMalloc((void**)&gpu_data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));

    cudaMemcpy(gpu_data,Data.data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    copyArrayKernel<<<gridSize,blockSize>>>(gpu_data,gpu_data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS));
    
    cudaDeviceSynchronize();

    cudaMemcpy(Data.data_cpy,gpu_data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyDeviceToHost);
    
    // printCSVData(data_cpy);
    cudaFree(gpu_data);
    cudaFree(gpu_data_cpy);
    return 1;
}

int calculateVarSquared(int xvar){
    int *gpu_data;
    int *gpu_result;
    int sum=0;

    cudaMalloc((void**)&gpu_data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));
    cudaMemcpy(gpu_data,Data.data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice);

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


int calculateSumOfProductSquared(int var1, int var2){
    int *gpuData,*gpuResult;
    int sum=0;

    cudaMalloc((void**)&gpuData,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));
    cudaMemcpy(gpuData,Data.data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice);

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

int calculateVarSum(int var){
    int *gpu_data;
    int *gpu_result;
    int sum=0;

    cudaMalloc((void**)&gpu_data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));
    cudaMemcpy(gpu_data,Data.data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_result,sizeof(int));
    cudaMemcpy(gpu_result,&sum,sizeof(int),cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    calculateVarSumKernel<<<gridSize,blockSize>>>(gpu_data,gpu_result,(MAX_VARIABLES*MAX_DATA_POINTS),var);

    cudaDeviceSynchronize();
    cudaMemcpy(&sum,gpu_result,sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(gpu_data);
    cudaFree(gpu_result);

    if (var==0) globalCalculationInfo.sumY=sum;
    if (var==1) globalCalculationInfo.sumX1=sum;
    if (var==2) globalCalculationInfo.sumX2=sum;
    return 0;
}

int predictModel(){
    int *gpuData;
    double *gpuPredictions,*gpuResiduals;

    cudaMalloc((void**)&gpuData,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int));
    cudaMalloc((void**)&gpuPredictions,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(double));
    cudaMalloc((void**)&gpuResiduals,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(double));

    cudaMemcpy(gpuData,Data.data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    predictModelKernel<<<gridSize,blockSize>>>(gpuData,gpuPredictions,gpuResiduals,globalRegressionInfo.beta_0,globalRegressionInfo.beta_1,globalRegressionInfo.beta_2,MAX_VARIABLES,Data.numObservations);

    cudaDeviceSynchronize();

    cudaMemcpy(Data.predictions,gpuPredictions,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(globalCalculationInfo.residuals,gpuResiduals,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(double),cudaMemcpyDeviceToHost);
    return 0;
}

int calculateStandardErrors(){
    return 0;
}
int calculateBetas(){
    double x1=globalCalculationInfo.sumSquaredX1-(pow(globalCalculationInfo.sumX1,2)/Data.numObservations);
    double x2=globalCalculationInfo.sumSquaredX2-(pow(globalCalculationInfo.sumX2,2)/Data.numObservations);
    double x1y=globalCalculationInfo.sumX1Y-(globalCalculationInfo.sumX1*globalCalculationInfo.sumY)/Data.numObservations;
    double x2y=globalCalculationInfo.sumX2Y-(globalCalculationInfo.sumX2*globalCalculationInfo.sumY)/Data.numObservations;
    double x1x2=globalCalculationInfo.sumX1X2-(globalCalculationInfo.sumX1*globalCalculationInfo.sumX2)/Data.numObservations;

    globalRegressionInfo.beta_1=((x2*x1y)-(x1x2*x2y))/((x1*x2)-pow(x1x2,2));
    globalRegressionInfo.beta_2=((x1*x2y)-(x1x2*x1y))/((x1*x2)-pow(x1x2,2));

    globalRegressionInfo.beta_0=(globalCalculationInfo.sumY/Data.numObservations)-
                                (globalRegressionInfo.beta_1*(globalCalculationInfo.sumX1/Data.numObservations))-
                                (globalRegressionInfo.beta_2*(globalCalculationInfo.sumX2/Data.numObservations));

    return 0;  
}

int runRegression(){
    printf("Running Regression...\n");
    calculateVarSquared(1);
    calculateVarSquared(2);
    calculateSumOfProductSquared(0,1);
    calculateSumOfProductSquared(0,2);
    calculateSumOfProductSquared(1,2);
    calculateVarSum(0);
    calculateVarSum(1);
    calculateVarSum(2);
    calculateBetas();
    predictModel();
    return 1;
}

int executeCommand(char command[]) {
    // exit
    if (strcmp(command, "e") == 0) {
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
            copyDataToGPU();
        } else {
            printf("No such CSV file exists AND/OR Error in reading CSV File\n");
        }
        return 1;
    }

    // view
    if (strcmp(command, "view") == 0) {
        if (data_primed) {
            printCSVData(Data.data);
            printf("Number of vars: %d\nNumber of observations: %d\n",Data.numVars,Data.numObservations);
        } else {
            printf("Data is not loaded yet\n");
        }
        return 1;
    }


    if (strcmp(command,"def")==0){
        data_primed=true;
        readCSV("csv.csv");
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

