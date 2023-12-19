/**
regression_info.cuh
--------------------------------------------------------------------------
DESCRPTION:
Define Structs used in regression.cu in this file
--------------------------------------------------------------------------
*/
#define MAX_VARIABLES 3
#define MAX_DATA_POINTS 100000
#define MAX_VARIABLE_NAME_LENGTH 50
#define MAX_COMMAND_LENGTH 100

// This struct stores all relevant "backend", calculation-side information.
typedef struct calculationInfo_t{
    double residuals[MAX_VARIABLES * MAX_DATA_POINTS];
    int sumSquaredX1;
    int sumSquaredX2;
    int sumX1X2;
    int sumX1Y;
    int sumX2Y;
    int sumX1;
    int sumX2;
    int sumY;
    double varianceY;
    double varianceX1;
    double varianceX2;
    double residualVariance;
} calculationInfo_t;

// This struct stores Beta Coefficients for each of the variables
typedef struct betaCoefficients_t{
    double beta_0;
    double beta_1;
    double beta_2;
} betaCoefficients_t;

// This struct stores the relevant standard errors for each of the variables
typedef struct standardErrors_t{
    double beta_0_stderr;
    double beta_1_stderr;
    double beta_2_stderr;
} standardErrors_t;

// This struct is the "data" that we use. Copy of the data, original data, and relevant data information is stored here
typedef struct data_t{
    int data[MAX_VARIABLES * MAX_DATA_POINTS];
    int data_cpy[MAX_VARIABLES * MAX_DATA_POINTS];
    char variableNames[MAX_VARIABLES][256];
    int numVars;
    int numObservations;
    

    int predictions[MAX_VARIABLES * MAX_DATA_POINTS];
} data_t;

// Coefficient p-values are stored in this struct
typedef struct pValues_t{
    double beta_0_pVal;
    double beta_1_pVal;
    double beta_2_pVal;
} pValues_t;

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
We store these indexes in this struct
*/
typedef struct varIndex_t{
    int zerothIndex;
    int firstIndex;
    int secondIndex;
} varIndex_t;

