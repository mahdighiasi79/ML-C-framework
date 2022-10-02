//
// Created by Mahdi on 5/2/2022.
//

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef C_CODE_ACTIVATION_FUNCTIONS_H
#define C_CODE_ACTIVATION_FUNCTIONS_H


double **sigmoid(double **x, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = 1 / (1 + exp(-x[i][j]));
    }
    return result;
}


double **tanH(double **x, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++) {
            double t = x[i][j];
            result[i][j] = (exp(t) - exp(-t)) / (exp(t) + exp(-t));
        }
    }
    return result;
}


double **relu(double **x, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++) {
            if (x[i][j] < 0)
                result[i][j] = 0;
            else
                result[i][j] = x[i][j];
        }
    }
    return result;
}


double **leaky_relu(double **x, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++) {
            if (x[i][j] < 0)
                result[i][j] = 0.01 * x[i][j];
            else
                result[i][j] = x[i][j];
        }
    }
    return result;
}


double **softMax(double **x, int row, int column) {

    double *reminder = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
    for (int i = 0; i < column; i++) {
        for (int j = 0; j < row; j++)
            reminder[i] += exp(x[j][i]);
    }

    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = exp(x[i][j]) / reminder[j];
    }
    return result;
}


double **d_softMax(double **x, int row, int column) {

    double *reminder = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
    for (int i = 0; i < column; i++) {
        for (int j = 0; j < row; j++)
            reminder[i] += exp(x[j][i]);
    }

    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = ((exp(x[i][j]) * reminder[j]) - exp(2 * x[i][j])) / pow(reminder[j], 2);
    }
    return result;
}


double **d_sigmoid(double **x, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++)
            result[i][j] = exp(-x[i][j]) / (1 + exp(-x[i][j]));
    }
    return result;
}


double **d_tanH(double **x, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++) {
            double t = x[i][j];
            result[i][j] = 1 - pow((exp(t) - exp(-t)) / (exp(t) + exp(-t)), 2);
        }
    }
    return result;
}


double **d_relu(double **x, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++)
            if (x[i][j] < 0)
                result[i][j] = 0;
            else
                result[i][j] = 1;
    }
    return result;
}


double **d_leaky_relu(double **x, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < column; j++)
            if (x[i][j] < 0)
                result[i][j] = 0.01;
            else
                result[i][j] = 1;
    }
    return result;
}


double **g(char *activation_function, double **x, int row, int column) {
    if (strcmp(activation_function, "sigmoid") == 0)
        return sigmoid(x, row, column);
    else if (strcmp(activation_function, "relu") == 0)
        return relu(x, row, column);
    else if (strcmp(activation_function, "leaky_relu") == 0)
        return leaky_relu(x, row, column);
    else if (strcmp(activation_function, "tanh") == 0)
        return tanH(x, row, column);
    else if (strcmp(activation_function, "softMax") == 0)
        return softMax(x, row, column);
    printf("not a valid activation function!");
    return NULL;
}


double **d_g(char *activation_function, double **x, int row, int column) {
    if (strcmp(activation_function, "sigmoid") == 0)
        return d_sigmoid(x, row, column);
    else if (strcmp(activation_function, "relu") == 0)
        return d_relu(x, row, column);
    else if (strcmp(activation_function, "leaky_relu") == 0)
        return d_leaky_relu(x, row, column);
    else if (strcmp(activation_function, "tanh") == 0)
        return d_tanH(x, row, column);
    else if (strcmp(activation_function, "softMax") == 0)
        return d_softMax(x, row, column);
    printf("not a valid activation function!");
    return NULL;
}


#endif //C_CODE_ACTIVATION_FUNCTIONS_H
