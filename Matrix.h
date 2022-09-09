//
// Created by Mahdi on 5/2/2022.
//

#include <stdlib.h>
#include <omp.h>

#ifndef C_CODE_MATRIX_H
#define C_CODE_MATRIX_H

double **matrix_multiplication(double **matrix1, double **matrix2, int row1, int column1, int column2) {

    double **result = (double **) malloc(row1 * sizeof(double *));
    for (int i = 0; i < column2; i++)
        result[i] = (double *) malloc(column2 * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < row1; i++) {

#pragma omp parallel for
        for (int j = 0; j < column2; j++) {
            for (int k = 0; k < column1; k++)
                result[i][j] += matrix1[i][k] * matrix2[k][j];
        }
    }

    return result;
}


double **matrix_addition(double **matrix1, double **matrix2, int row, int column) {

    double **result = (double **) malloc(row * sizeof(double *));
    for (int i = 0; i < row; i++)
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = matrix1[i][j] + matrix2[i][j];
    }

    return result;
}

#endif //C_CODE_MATRIX_H
