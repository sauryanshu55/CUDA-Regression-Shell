/**
regression.cu
--------------------------------------------------------------------------
DESCRPTION:
This is the main file that compiles into the shell.
We read CSV from this file, implment shell commands in this file.
We define wrappers for the parallel CUDA code, in kernels.cuh 
--------------------------------------------------------------------------
*/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include<unistd.h>
#include <math.h>
#include <gsl/gsl_cdf.h>
#include "kernels.cuh"
#include "regression_info.cuh"

#define MAX_VARIABLES 3 //Max number of variables
#define MAX_DATA_POINTS 1000 // Max number of data points
#define MAX_VARIABLE_NAME_LENGTH 50 //Max variable name length
#define MAX_COMMAND_LENGTH 100 //maximum length of command

// Data is stored globally on the data variable
data_t Data;

// Indicates if data has been loaded from the csv file yet
bool data_primed = false;

// Store Calc info (See struct comments)
calculationInfo_t globalCalculationInfo;

// Store beta coefficients
betaCoefficients_t betaCoefficients;

// Store standard errors
standardErrors_t standardErrors;

// Store P Values
pValues_t pValues;

// Indexes are assigned according to the "reg [var1] [var2] [var3]"
// See struct definition. Zeroth element of the struct is assigned to the index of var1
// First element of the struct is assigned to the index of var2 and so on
varIndex_t indexes;

/*
P: int* data (The Data to be passed to print)
Takes in the Data from the CSV file and print it in an orderly fashion
*/
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

/*
P: const char *filename (the location of the csv file)
Read the file csv file and store it in a 1-D array
*/
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
            token = strtok(NULL, ",");
            variableIndex++;
        }

        dataPointIndex++;
    }
    Data.numObservations = dataPointIndex;

    fclose(file);
    return 0;
}

/*
Based on the regression parameters provided,use parallel computation to make a data copy to be used in the regressions, into an array.
For example:
    Data is initally loaded as:
        y x1 x2
        0  1  2 : Original Indexes
    A copy of the Data is converted to the following, given the user command:
        reg x2 y x1, then
        x2 y x1
        2  0  1
*/
int createdIndexedData(){
    int *gpu_data, *gpu_data_cpy; // gpu_data: Data to be sent to GPU. gpu_data_cpy: Output from GPU
    
    // Malloc space for original data
    if(cudaMalloc((void**)&gpu_data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Malloc space for GPU output
    if (cudaMalloc((void**)&gpu_data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Copy Original Data to GPU
    if (cudaMemcpy(gpu_data,Data.data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Set block and gridsize
    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    // Call Kernels
    createdIndexedDataKernel<<<gridSize,blockSize>>>(gpu_data,gpu_data_cpy,0,indexes.zerothIndex,(MAX_VARIABLES*MAX_DATA_POINTS));
    createdIndexedDataKernel<<<gridSize,blockSize>>>(gpu_data,gpu_data_cpy,1,indexes.firstIndex,(MAX_VARIABLES*MAX_DATA_POINTS));
    createdIndexedDataKernel<<<gridSize,blockSize>>>(gpu_data,gpu_data_cpy,2,indexes.secondIndex,(MAX_VARIABLES*MAX_DATA_POINTS));
    
    // Synchronize Device
    if (cudaDeviceSynchronize()!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Copy back results from GPU to Host
    if (cudaMemcpy(Data.data_cpy,gpu_data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    cudaFree(gpu_data);
    cudaFree(gpu_data_cpy);

    return 1;
}

/*
Calculate square of the given variable paralelly
P: xvar(0 if y, 1 if x1, 2 if x2)
*/
int calculateVarSquared(int xvar){
    int *gpu_data; // Store data in GPU
    int *gpu_result; // Store gpu result
    int sum=0; 

    // Malloc space for gpu data
    if(cudaMalloc((void**)&gpu_data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    
    // Send result array for gpu result array (need an array becasue atomicAdd needs to be passed an array for atomic addition)
    if(cudaMemcpy(gpu_data,Data.data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Malloc space for GPU result 
    if(cudaMalloc((void**)&gpu_result,sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Copy result from host to device
    if(cudaMemcpy(gpu_result,&sum,sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    calculateSquareKernel<<<gridSize,blockSize>>>(gpu_data,gpu_result,(MAX_VARIABLES*MAX_DATA_POINTS),xvar);

    // Sync GPU and host
    if (cudaDeviceSynchronize()!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    
    // Copy Result back
    if (cudaMemcpy(&sum,gpu_result,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    cudaFree(gpu_data);
    cudaFree(gpu_result);

    // Store into appropriate struct  
    if(xvar==1)  globalCalculationInfo.sumSquaredX1=sum;
    if(xvar==2) globalCalculationInfo.sumSquaredX2=sum;

    return 0;
}

/**
Calculate sum of product squared parallely
P: var1 (first var index)
   var2 (second var index)
*/
int calculateSumOfProductSquared(int var1, int var2){
    int *gpuData,*gpuResult; // Variable to GPU data and result
    int sum=0;

    // Malloc space for GPU and copy it to GPU
    // We pass gpuResult, an array, since we need to pass it for atomicAdd function.
    if(cudaMalloc((void**)&gpuData,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMemcpy(gpuData,Data.data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMalloc((void**)&gpuResult,sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMemcpy(gpuResult,&sum,sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Set grid and block size
    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    calculateSumOfProductSquaredKernel<<<gridSize,blockSize>>>(gpuData,gpuResult,(MAX_VARIABLES*MAX_DATA_POINTS),var1,var2);
    
    // Syncrhronize GPU and host
    if(cudaDeviceSynchronize()!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Get result back from GPU
    if(cudaMemcpy(&sum,gpuResult,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    cudaFree(gpuResult);
    cudaFree(gpuData);

    // Assign to appropriate struct
    if ((var1==0)&&(var2==1)) globalCalculationInfo.sumX1Y=sum;
    if ((var1==0)&&(var2==2)) globalCalculationInfo.sumX2Y=sum;
    if ((var1==1)&&(var2==2)) globalCalculationInfo.sumX1X2=sum;

    return 0;
}

/*
Calculate Variable Sum parallely
P: var (the variable index to calculate it for)
*/
int calculateVarSum(int var){
    // Initialize vars to send to GPU
    int *gpu_data;
    int *gpu_result;
    int sum=0;

    // Malloc space and copy stuff to GPU     
    if(cudaMalloc((void**)&gpu_data,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMemcpy(gpu_data,Data.data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMalloc((void**)&gpu_result,sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    
    if(cudaMemcpy(gpu_result,&sum,sizeof(int),cudaMemcpyHostToDevice)){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Block and GridSize
    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    // Call Kernel
    calculateVarSumKernel<<<gridSize,blockSize>>>(gpu_data,gpu_result,(MAX_VARIABLES*MAX_DATA_POINTS),var);

    // Synchronize Device
    if(cudaDeviceSynchronize()!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    
    // Copy Result back
    if(cudaMemcpy(&sum,gpu_result,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    cudaFree(gpu_data);
    cudaFree(gpu_result);

    // Assign to appropriate struct
    if (var==0) globalCalculationInfo.sumY=sum;
    if (var==1) globalCalculationInfo.sumX1=sum;
    if (var==2) globalCalculationInfo.sumX2=sum;
    return 0;
}

/*
Predict Model paralelly from calculaed GPU
*/
int predictModel(){
    // Initialize Variables to send to  GPU
    int *gpuData;
    int *gpuPredictions;
    int *gpuResiduals;

    // Malloc space for data in GPU and copy it to the GPU
    if(cudaMalloc((void**)&gpuData,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    if(cudaMalloc((void**)&gpuPredictions,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    if(cudaMalloc((void**)&gpuResiduals,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMemcpy(gpuData,Data.data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Set Grid size and blocksize
    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    // Call Kernel
    predictModelKernel<<<gridSize,blockSize>>>(gpuData,gpuPredictions,gpuResiduals,betaCoefficients.beta_0,betaCoefficients.beta_1,betaCoefficients.beta_2,MAX_VARIABLES,Data.numObservations);

    // Synchronize Device with GPU
    if(cudaDeviceSynchronize()!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Copy results GPU
    if(cudaMemcpy(Data.predictions,gpuPredictions,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Copy results GPU
    if(cudaMemcpy(globalCalculationInfo.residuals,gpuResiduals,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    

    cudaFree(gpuData);
    cudaFree(gpuPredictions);
    cudaFree(gpuResiduals);
    return 0;
}

/*
Calculate the standard errors for the residuals of the model parallely
*/
int calculateStandardErrors(){
    // Initialize values for GPU
    int *gpuResiduals, *gpuResidualSum;
    int *gpuVarianceArr;

    double *gpuVarianceResiduals;
    double varianceResiduals =0;

    int sum=0;

    //  Malloc host code to GPU
    if(cudaMalloc((void**)&gpuVarianceResiduals,sizeof(double))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    
    if(cudaMemcpy(gpuVarianceResiduals,&varianceResiduals,sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMalloc((void**)&gpuResiduals,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMemcpy(gpuResiduals,globalCalculationInfo.residuals,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMalloc((void**)&gpuResidualSum,sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    
    if(cudaMemcpy(gpuResidualSum,&sum,sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMalloc((void**)&gpuVarianceArr,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    // Call Kernel 
    calculateStandardErrorsKernel<<<gridSize,blockSize>>>(gpuResiduals,gpuResidualSum,gpuVarianceArr,gpuVarianceResiduals,Data.numObservations,(MAX_VARIABLES*MAX_DATA_POINTS));

    // Synchrnonize Device 
    if(cudaDeviceSynchronize()!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    
    // Copy back results
    if(cudaMemcpy(&sum,gpuResidualSum,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Copy back results
    if(cudaMemcpy(&varianceResiduals,gpuVarianceResiduals,sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Assign result to struct
    globalCalculationInfo.residualVariance=varianceResiduals;

    cudaFree(gpuVarianceResiduals);
    cudaFree(gpuResidualSum);
    cudaFree(gpuResiduals);

    return 0;
}

/**
Calculate Variance for the given variables parallely
*/
int calculateVarVariance(int var){
    // Initialize GPU data
    int *gpuData, *gpuNumeratorSumArr;
    double *gpuVarVariance;

    double variance=0;

    // Malloc and copy data from Host to Device
    if(cudaMalloc((void**)&gpuData,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    if(cudaMemcpy(gpuData,Data.data_cpy,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMalloc((void**)&gpuVarVariance,sizeof(double))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    
    if(cudaMemcpy(gpuVarVariance,&variance,sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    if(cudaMalloc((void**)&gpuNumeratorSumArr,(MAX_VARIABLES*MAX_DATA_POINTS)*sizeof(int))!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Set grid and blocksize
    int blockSize = 256;
    int gridSize = ((MAX_VARIABLES*MAX_DATA_POINTS)+ blockSize - 1) / blockSize;

    // Calculate mean, to be passed to kernel
    int varMean = 0;
    if (var==0) varMean=globalCalculationInfo.sumY/Data.numObservations;
    if (var==1) varMean=globalCalculationInfo.sumX1/Data.numObservations;
    if (var==2) varMean=globalCalculationInfo.sumX2/Data.numObservations;

    calculateVarVarianceKernel<<<gridSize,blockSize>>>(gpuData,gpuNumeratorSumArr,gpuVarVariance,
                                                        Data.numObservations,var,varMean,
                                                        (MAX_VARIABLES*MAX_DATA_POINTS));

    // Syncrhonize Device
    if(cudaDeviceSynchronize()!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }
    
    // Copy back result from GPU to Host
    if(cudaMemcpy(&variance,gpuVarVariance,sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess){
        fprintf(stderr,"Failed to allocate memory to GPU\n");
        exit(-1);
    }

    // Assign to apppropirate struct
    if (var==0) globalCalculationInfo.varianceY=variance;
    if (var==1) globalCalculationInfo.varianceX1=variance;
    if (var==2) globalCalculationInfo.varianceX2=variance;

    cudaFree(gpuData);
    cudaFree(&gpuNumeratorSumArr);
    cudaFree(&gpuVarVariance);

    return 0;
}

/*
This function uses all the relevant info calculated till now and calculated the beta coefficients
*/
int calculateBetas(){
    calculationInfo_t cInfo=globalCalculationInfo;

    double x1=cInfo.sumSquaredX1-(pow(cInfo.sumX1,2)/Data.numObservations);
    double x2=cInfo.sumSquaredX2-(pow(cInfo.sumX2,2)/Data.numObservations);
    double x1y=cInfo.sumX1Y-(cInfo.sumX1*cInfo.sumY)/Data.numObservations;
    double x2y=cInfo.sumX2Y-(cInfo.sumX2*cInfo.sumY)/Data.numObservations;
    double x1x2=cInfo.sumX1X2-(cInfo.sumX1*cInfo.sumX2)/Data.numObservations;

    betaCoefficients.beta_1=((x2*x1y)-(x1x2*x2y))/((x1*x2)-pow(x1x2,2));
    betaCoefficients.beta_2=((x1*x2y)-(x1x2*x1y))/((x1*x2)-pow(x1x2,2));

    betaCoefficients.beta_0=(cInfo.sumY/Data.numObservations)-
                                (betaCoefficients.beta_1*(cInfo.sumX1/Data.numObservations))-
                                (betaCoefficients.beta_2*(cInfo.sumX2/Data.numObservations));

    return 0;  
}

/*
Calculate P values using GSL libraries
*/
void calculatePValues(){
    double t_Stat_beta_0=betaCoefficients.beta_0/standardErrors.beta_0_stderr;
    double t_Stat_beta_1=betaCoefficients.beta_1/standardErrors.beta_1_stderr;
    double t_Stat_beta_2=betaCoefficients.beta_0/standardErrors.beta_2_stderr;

    int degreesOfFreedom=Data.numObservations-3;

    pValues.beta_0_pVal=2*(1-gsl_cdf_tdist_P(fabs(t_Stat_beta_0),degreesOfFreedom));
    pValues.beta_1_pVal=2*(1-gsl_cdf_tdist_P(fabs(t_Stat_beta_1),degreesOfFreedom));
    pValues.beta_2_pVal=2*(1-gsl_cdf_tdist_P(fabs(t_Stat_beta_2),degreesOfFreedom));
    
}

/*
Print Regression Results in an orderly manner
*/
int printRegressionResults(){
    size_t boundarySize=70;
    char boundary[boundarySize];
    memset(boundary,'-',boundarySize-1);
    boundary[boundarySize-1]='\0';

    printf("\nOutput: %s\n%s\n",Data.variableNames[indexes.zerothIndex],boundary);
    printf("Var         Coeff         Variance         StdErr         P(|T|>t)\n %s\n",boundary);
    printf("cons    %lf    %lf    %lf    %lf\n",betaCoefficients.beta_0,globalCalculationInfo.varianceY,standardErrors.beta_0_stderr,pValues.beta_0_pVal);
    printf("%s      %lf    %lf    %lf    %lf\n",Data.variableNames[indexes.firstIndex],betaCoefficients.beta_1,globalCalculationInfo.varianceX1,standardErrors.beta_1_stderr,pValues.beta_1_pVal);
    printf("%s      %lf    %lf    %lf    %lf\n\n",Data.variableNames[indexes.secondIndex],betaCoefficients.beta_2,globalCalculationInfo.varianceX2,standardErrors.beta_2_stderr,pValues.beta_2_pVal);
    return 0;
}

/*
Run Regression and all the required components.
Call all kernel wrappers to call GPU code
*/
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

    calculateStandardErrors();
    
    calculateVarVariance(0);
    calculateVarVariance(1);
    calculateVarVariance(2);

    standardErrors.beta_0_stderr=sqrt(globalCalculationInfo.residualVariance*globalCalculationInfo.varianceY);
    standardErrors.beta_1_stderr=sqrt(globalCalculationInfo.residualVariance*globalCalculationInfo.varianceX1);
    standardErrors.beta_2_stderr=sqrt(globalCalculationInfo.residualVariance*globalCalculationInfo.varianceX2);

    calculatePValues();

    printRegressionResults();
    return 1;
}

/*
Based on the regression parameters provided,use parallel computation to make a data copy to be used in the regressions, into an array.
For example:
    Data is initally loaded as:
        y x1 x2
        0  1  2 : Original Indexes
    A copy of the Data is converted to the following, given the user command:
        reg x2 y x1, then
        x2 y x1
        2  0  1
*/
void assignVariableIndexes(char* givenY, char*  givenX1, char* givenX2){
    // Go through all variables, and the user given variables. Assign actual variables accordingly
    if (strcmp(Data.variableNames[0],givenY)==0) indexes.zerothIndex=0;
    if (strcmp(Data.variableNames[1],givenY)==0) indexes.zerothIndex=1;
    if (strcmp(Data.variableNames[2],givenY)==0) indexes.zerothIndex=2;

    if (strcmp(Data.variableNames[0],givenX1)==0) indexes.firstIndex=0;
    if (strcmp(Data.variableNames[1],givenX1)==0) indexes.firstIndex=1;
    if (strcmp(Data.variableNames[2],givenX1)==0) indexes.firstIndex=2;

    if (strcmp(Data.variableNames[0],givenX2)==0) indexes.secondIndex=0;
    if (strcmp(Data.variableNames[1],givenX2)==0) indexes.secondIndex=1;
    if (strcmp(Data.variableNames[2],givenX2)==0) indexes.secondIndex=2;
}

/**
Check provided Regression parameters. If they donot match data variables, return false 
*/
int checkRegressionParameters(char* givenY, char*  givenX1, char* givenX2){
    // printf("%s %s %s\n",givenY,givenX1,givenX2);
    // printf("%s %s %s\n",Data.variableNames[0],Data.variableNames[1],Data.variableNames[2]);
    // printf("%d\n",strcmp(Data.variableNames[0],givenY));
    // printf("%d\n",strcmp(Data.variableNames[1],givenX1));
    // printf("%d\n",strcmp(Data.variableNames[2],givenX2));
    for (int i=0;i<=2;i++){
        char* comparator=Data.variableNames[i];
        if (
            (strcmp(comparator,givenY)!=0) &&
            (strcmp(comparator,givenX1)!=0) &&
            (strcmp(comparator,givenX2)!=0)
        ){
            return false;
        }
    }
    return true;
}

// Execute shell commands
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

    //  reg command
    if (strncmp(command, "reg ", 4) == 0) {
        if (data_primed){
            char givenY[MAX_VARIABLE_NAME_LENGTH], givenX1[MAX_VARIABLE_NAME_LENGTH], givenX2[MAX_VARIABLE_NAME_LENGTH];  
        // Extract the next three variables
            int offset = 4;  // Skip the "reg " part
            int count = sscanf(command + offset, "%49s %49s %49s", givenY, givenX1, givenX2);

            if ((count == 3) && (checkRegressionParameters(givenY, givenX1, givenX2))) {
                assignVariableIndexes(givenY, givenX1, givenX2);
                createdIndexedData();
                runRegression();
                return 1;
            } else {
                printf("Invalid regression paramter specified\n");
                return 1;
            }
        } else {
            printf("Data is not loaded yet.\n");
            return 1;
        }

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

