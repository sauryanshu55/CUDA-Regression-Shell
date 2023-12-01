#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"

#define MAX_VARIABLES 50
#define MAX_DATA_POINTS 1000
#define MAX_VARIABLE_NAME_LENGTH 50
#define MAX_COMMAND_LENGTH 100

// Global array to store data
double data[MAX_VARIABLES][MAX_DATA_POINTS];
// Array to store variable names
char variableNames[MAX_VARIABLES][MAX_VARIABLE_NAME_LENGTH];
// Matrix
double **matrix;

int numVars;
int numObservations;
bool data_primed = false;

// Function to populate the matrix with data
void createDataMatrix() {
        // Allocate memory for the matrix
    matrix = (double **)malloc((numObservations) * sizeof(double *));
    for (int i = 0; i < numObservations; i++) {
        matrix[i] = (double *)malloc((numVars) * sizeof(double)); // +1 for the additional column
    }

    // Populate the matrix with 1's in the first column and data in the rest
    for (int i = 0; i < numObservations; i++) {
        matrix[i][0] = 1.0; // Populate the first column with 1's

        for (int j = 1; j <= numVars+1; j++) {
            matrix[i][j] = data[j][i]; // Populate the rest of the columns with data
        }
    }
}

void printDataMatrix() {
    for (int i = 0; i < numObservations; i++) {
        for (int j = 0; j < numVars; j++) {
            printf("%.2f\t", matrix[i][j]);
        }
        printf("\n");
    }
}

// Function to print the imported data
void printCSVData() {
    printf("Variable Names:\n");
    for (int i = 0; i < MAX_VARIABLES && variableNames[i][0] != '\0'; i++) {
        printf("%s\t", variableNames[i]);
    }

    printf("\n");

    for (int j = 0; j < MAX_DATA_POINTS && data[0][j] != 0; j++) {
        for (int i = 0; i < MAX_VARIABLES && variableNames[i][0] != '\0'; i++) {
            printf("%lf\t", data[i][j]);
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
        numVars=variableIndex;
    }

    // Read the rest of the file to get numerical data
    int dataPointIndex = 0;
    while (fgets(line, sizeof(line), file) != NULL && dataPointIndex < MAX_DATA_POINTS) {
        char *token = strtok(line, ",");
        int variableIndex = 0;

        while (token != NULL && variableIndex < MAX_VARIABLES) {
            sscanf(token, " %lf", &data[variableIndex][dataPointIndex]);

            // Print data if needed
            // printf("%s: %lf\n", variableNames[variableIndex], data[variableIndex][dataPointIndex]);

            token = strtok(NULL, ",");
            variableIndex++;
        }

        dataPointIndex++;
    }
    numObservations=dataPointIndex-1;

    fclose(file);
    return 0;
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
            createDataMatrix();
        } else {
            printf("No such CSV file exists AND/OR Error in reading CSV File\n");
        }
        return 1;
    }

    // view
    if (strcmp(command, "view") == 0) {
        if (data_primed) {
            printCSVData();
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

        double **test=createMatrix(5,5);
        double **dest=copyMatrix(test,5,5);
        printMatrix(dest,5,5);
        
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

