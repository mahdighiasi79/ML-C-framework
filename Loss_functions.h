//
// Created by mmghf on 10/1/2022.
//

#ifndef C_CODE_LOSS_FUNCTIONS_H
#define C_CODE_LOSS_FUNCTIONS_H


#include <string.h>

#include "Matrix.h"

double SSE(double **output, double **labels, int row, int column) {
    double **loss = matrix_subtraction(output, labels, row, column);
    loss = power_matrix(loss, 2, row, column);
    loss = sum_axis(loss, 0, row, column);
    loss = sum_axis(loss, 1, 1, column);
    double result = loss[0][0];
    result /= column;
    return result;
}


double cross_entropy(double **output, double **labels, int row, int column) {
    double **log_y_h = log_matrix(output, row, column);
    double **y_log_y_h = element_wise_mul(labels, log_y_h, row, column);
    double **loss = sum_axis(y_log_y_h, 0, row, column);
    loss = sum_axis(loss, 1, 1, column);
    double result = loss[0][0];
    result /= -column;
    return result;
}


double **d_cross_entropy(double **output, double **labels, int row, int column) {
    double **result = element_wise_div(labels, output, row, column);
    result = matrix_constant_mul(result, -1, row, column);
    return result;
}


double **d_SSE(double **output, double **labels, int row, int column) {
    double **result = matrix_subtraction(output, labels, row, column);
    result = matrix_constant_mul(result, 2, row, column);
    return result;
}


double loss(char *loss_function, double **output, double **labels, int row, int column) {
    if (strcmp(loss_function, "SSE") == 0)
        return SSE(output, labels, row, column);
    else if (strcmp(loss_function, "cross_entropy") == 0)
        return cross_entropy(output, labels, row, column);
    printf("invalid loss function!");
    return -1;
}


double **d_loss(char *loss_function, double **output, double **labels, int row, int column) {
    if (strcmp(loss_function, "SSE") == 0)
        return d_SSE(output, labels, row, column);
    else if (strcmp(loss_function, "cross_entropy") == 0)
        return d_cross_entropy(output, labels, row, column);
    printf("invalid loss function!");
    return NULL;
}


#endif //C_CODE_LOSS_FUNCTIONS_H
