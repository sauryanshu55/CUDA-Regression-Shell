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

// Function for multiplying 2 matrices
double **multiplyMatrix(double ** mat1, double **mat2, int rows1, int cols1, int rows2, int cols2) {
    double **result = createMatrix(rows,columns)
    if (cols1 != rows2) {
        printf("Matrix dimensions do not allow multiplication. Do better\n");
        return;
    }
    // store the multiplication in result by mutipliying each element in each array
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result; 
}

// Function to calculate the transpose of a matrix
double** transpose(double** matrix, int rows, int columns) {
    double** result = createMatrix(columns, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}


// Function to calculate the determinant of a matrix
double determinant(double** matrix, int rows, int columns) {
    if (rows != columns) {
        printf("Matrix is not square. Determinant cannot be calculated.\n");
        return 0.0;
    }

    double det = 0;
    if (rows == 1) {
        return matrix[0][0];
    }
    else if (rows == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    else {
        double** temp = createMatrix(rows, columns);
        int sign = 1;
        for (int f = 0; f < rows; f++) {
            int i = 0, j = 0;
            for (int row = 1; row < rows; row++) {
                for (int col = 0; col < columns; col++) {
                    if (col != f) {
                        temp[i][j++] = matrix[row][col];
                        if (j == rows - 1) {
                            j = 0;
                            i++;
                        }
                    }
                }
            }
            det += sign * matrix[0][f] * determinant(temp, rows - 1, columns - 1);
            sign = -sign;
        }
        freeMatrix(temp, rows);
    }
    return det;
}

// Function to calculate the cofactor of a matrix
void getCofactor(double** matrix, int rows, int columns, double** temp, int row, int col, int p, int q) {
    int i = 0, j = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns; c++) {
            if (r != p && c != q) {
                temp[i][j++] = matrix[r][c];
                if (j == rows - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

// Function to calculate the inverse of a matrix
double** inverse(double** matrix, int rows, int columns) {
    double** result = createMatrix(rows, columns);
    double** temp = createMatrix(rows, columns);

    int n = rows; // Assuming square matrix
    double det = determinant(matrix, rows, columns, n);
    if (det == 0) {
        printf("Inverse doesn't exist for this matrix.\n");
        freeMatrix(result, rows);
        freeMatrix(temp, rows);
        return NULL;
    }

    double sign = 1.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            getCofactor(matrix, i, j, temp, rows, columns, i, j, n);
            sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            result[j][i] = (sign) * (determinant(temp, rows, columns, n - 1)) / det;
        }
    }

    double** transpose_result = transpose(result, rows, columns);
    return transpose_result;
}

