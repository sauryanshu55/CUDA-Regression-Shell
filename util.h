#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to create a 2D matrix filled with 0s
double **createMatrix(int rows, int columns) {
    // Allocate memory for the matrix
    double **matrix = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double *)malloc(columns * sizeof(double));
        // Initialize elements to 0
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = 0.0;
        }
    }
    return matrix;
}

// Function to print the content of a 2D matrix
void printMatrix(double **matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            printf("%lf\t", matrix[i][j]);
        }
        printf("\n");
    }
}

// Function to copy the content of a source matrix and return a new matrix
double **copyMatrix(double **source, int rows, int columns) {
    double **destination = createMatrix(rows, columns);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            destination[i][j] = source[i][j];
        }
    }
    return destination;
}