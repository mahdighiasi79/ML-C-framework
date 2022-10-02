#ifndef C_MATRIX_H
#define C_MATRIX_H


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


double **matrix_multiplication(double **matrix1, double **matrix2, int row1, int row2, int column2) {

    double **result = (double **) malloc(row1 * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row1; i++) {
        result[i] = (double *) malloc(column2 * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column2; j++) {
            result[i][j] = 0;

            for (int k = 0; k < row2; k++)
                result[i][j] += (matrix1[i][k] * matrix2[k][j]);
        }
    }

    return result;
}


double **matrix_broadcast_addition(double **matrix, double **b, int row, int column) {

    double **result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = matrix[i][j] + b[i][0];
    }

    return result;
}


double **matrix_addition(double **matrix1, double **matrix2, int row, int column) {

    double **result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = matrix1[i][j] + matrix2[i][j];
    }

    return result;
}


double **matrix_subtraction(double **matrix1, double **matrix2, int row, int column) {

    double **result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = matrix1[i][j] - matrix2[i][j];
    }

    return result;
}


double **matrix_constant_mul(double **matrix, double constant, int row, int column) {

    double **result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = constant * matrix[i][j];
    }

    return result;
}


double **matrix_constant_add(double **matrix, double constant, int row, int column) {

    double **result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = constant + matrix[i][j];
    }

    return result;
}


double **element_wise_mul(double **matrix1, double **matrix2, int row, int column) {

    double **result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = matrix1[i][j] * matrix2[i][j];
    }

    return result;
}


double **element_wise_div(double **matrix1, double **matrix2, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = matrix1[i][j] / matrix2[i][j];
    }

    return result;
}


double **transpose(double **matrix, int row, int column) {

    double **result = (double **) malloc(column * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < column; i++) {
        result[i] = (double *) malloc(row * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < row; j++)
            result[i][j] = matrix[j][i];
    }

    return result;
}


double **sum_axis(double **matrix, int axis, int row, int column) {

    double **result;

    if (axis == 0) {
        result = (double **) malloc(sizeof(double *));
        result[0] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int i = 0; i < column; i++) {

            for (int j = 0; j < row; j++)
                result[0][i] += matrix[j][i];
        }

    } else if (axis == 1) {
        result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
        for (int i = 0; i < row; i++) {
            result[i] = (double *) malloc(sizeof(double));

            for (int j = 0; j < column; j++)
                result[i][0] += matrix[i][j];
        }

    } else
        result = NULL;

    return result;
}


double **copy_matrix(double **matrix, int from, int to, int column) {
    double **result = (double **) malloc((to - from) * sizeof(double *));

#pragma omp parallel for
    for (int i = from; i < to; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = matrix[i][j];
    }
    return result;
}


int *arg_max(double **matrix, int axis, int row, int column) {
    int *result;

    if (axis == 0) {
        result = (int *) malloc(row * sizeof(int));

#pragma omp parallel for
        for (int i = 0; i < row; i++) {
            double biggest = -9007199254740992;
            int index = -1;
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] > biggest) {
                    index = j;
                    biggest = matrix[i][j];
                }
            }
            result[i] = index;
        }

    } else if (axis == 1) {
        result = (int *) malloc(column * sizeof(int));

#pragma omp parallel for
        for (int i = 0; i < column; i++) {
            double biggest = -9007199254740992;
            int index = -1;
            for (int j = 0; j < row; j++) {
                if (matrix[j][i] > biggest) {
                    biggest = matrix[j][i];
                    index = j;
                }
            }
            result[i] = index;
        }
    } else
        result = NULL;

    return result;
}


double **power_matrix(double **matrix, double p, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = pow(matrix[i][j], p);
    }
    return result;
}


double **shuffle(double **matrix, int m) {
    time_t t = time(NULL);
    srand(t);
    int r = floor((double) m / 4);

    for (int i = 0; i < r; i++) {
        int index1 = rand() % m;
        int index2 = rand() % m;
        double *temp = matrix[index1];
        matrix[index1] = matrix[index2];
        matrix[index2] = temp;
    }

    return matrix;
}


double **log_matrix(double **matrix, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = log(matrix[i][j]);
    }
    return result;
}


void print_matrix(double **matrix, int row, int column) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++)
            printf("%f ", matrix[i][j]);
        printf("\n");
    }
}


double normal_distribution(double x, double mean, double standard_deviation) {
    double result = 1 / (standard_deviation * pow(2 * M_PI, 0.5));
    result *= exp(-0.5 * pow((x - mean) / standard_deviation, 2));
    return result;
}


#endif //C_MATRIX_H