//
// Created by Mahdi on 5/2/2022.
//


#include <string.h>
#include <time.h>

#include "dataTypes.h"
#include "Matrix.h"
#include "Activation_functions.h"
#include "Loss_functions.h"

#ifndef C_CODE_DNN_H
#define C_CODE_DNN_H


Layer Dense(int units, char *activation_function) {
    Layer layer;
    layer.type = "Dense";
    layer.units = units;
    layer.activation_function = activation_function;
    layer.w = (double **) malloc(units * sizeof(double *));
    layer.b = (double **) malloc(units * sizeof(double *));
    layer.Vdw = (double **) malloc(units * sizeof(double *));
    layer.Vdb = (double **) malloc(units * sizeof(double *));
    layer.Sdw = (double **) malloc(units * sizeof(double *));
    layer.Sdb = (double **) malloc(units * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < units; i++) {
        layer.b[i] = (double *) malloc(sizeof(double));
        layer.Vdb[i] = (double *) malloc(sizeof(double));
        layer.Sdb[i] = (double *) malloc(sizeof(double));
    }
    return layer;
}


void initialize_model(Layer *model, int layers, int input_number) {
    time_t t = time(NULL);
    srand(t);

    if (strcmp(model[0].type, "Dense") == 0) {
#pragma omp paralle for
        for (int i = 0; i < model[0].units; i++) {
            model[0].w[i] = (double *) malloc(input_number * sizeof(double));
            model[0].Vdw[i] = (double *) malloc(input_number * sizeof(double));
            model[0].Sdw[i] = (double *) malloc(input_number * sizeof(double));
#pragma omp parallel for
            for (int j = 0; j < input_number; j++)
                model[0].w[i][j] = normal_distribution((double) rand(), 0, 1);
        }
    }

#pragma omp parallel for
    for (int i = 1; i < layers; i++) {
        if (strcmp(model[i].type, "Dense") == 0) {
#pragma omp parallel for
            for (int j = 0; j < model[i].units; j++) {
                model[i].w[j] = (double *) malloc(model[i - 1].units * sizeof(double));
                model[i].Vdw[j] = (double *) malloc(model[i - 1].units * sizeof(double));
                model[i].Sdw[j] = (double *) malloc(model[i - 1].units * sizeof(double));
#pragma omp paralle for
                for (int k = 0; k < model[i - 1].units; k++)
                    model[i].w[j][k] = normal_distribution((double) rand(), 0, 1);
            }
        }
    }
}


void forward_prop(Layer *model, int layers, double **inputs, int input_number, int m) {
    double **w_inputs;
    double **z;
    double **a;

    if (strcmp(model[0].type, "Dense") == 0) {
        w_inputs = matrix_multiplication(model[0].w, inputs, model[0].units, input_number, m);
        z = matrix_broadcast_addition(w_inputs, model[0].b, model[0].units, m);
        a = g(model[0].activation_function, z, model[0].units, m);
        model[0].z = z;
        model[0].a = a;
    }

    for (int i = 1; i < layers; i++) {
        if (strcmp(model[i].type, "Dense") == 0) {
            w_inputs = matrix_multiplication(model[i].w, model[i - 1].a, model[i].units, model[i - 1].units, m);
            z = matrix_broadcast_addition(w_inputs, model[i].b, model[i].units, m);
            a = g(model[i].activation_function, z, model[i].units, m);
            model[i].z = z;
            model[i].a = a;
        }
    }
}


void
back_prop(Layer *model, int layers, char *loss_function, double **inputs, double **labels, int input_number, int m) {

    double **g_prime_z;
    double **a_prev_t;
    double **w_next_t;
    double **g_prime_z_next;
    double **g_prime_z_next_ElementWiseMul_a_next;
    double **da;
    double **db;
    double **dw;
    double constant = 1 / (double) m;

    if (layers > 1) {
        g_prime_z = d_g(model->activation_function, model[layers - 1].z, model[layers - 1].units, m);
        a_prev_t = transpose(model[layers - 2].a, model[layers - 2].units, m);
        da = d_loss(loss_function, model[layers - 1].a, labels, model[layers - 1].units, m);
        db = element_wise_mul(da, g_prime_z, model[layers - 1].units, m);
        dw = matrix_multiplication(db, a_prev_t, model[layers - 1].units, m, model[layers - 2].units);
        dw = matrix_constant_mul(dw, constant, model[layers - 1].units, model[layers - 2].units);
        db = sum_axis(db, 1, model[layers - 1].units, m);
        db = matrix_constant_mul(db, constant, model[layers - 1].units, 1);
        model[layers - 1].da = da;
        model[layers - 1].dw = dw;
        model[layers - 1].db = db;

        for (int i = layers - 2; i > 0; i--) {
            w_next_t = transpose(model[i + 1].w, model[i + 1].units, model[i].units);
            g_prime_z_next = d_g(model->activation_function, model[i + 1].z, model[i + 1].units, model[i].units);
            g_prime_z_next_ElementWiseMul_a_next = element_wise_mul(g_prime_z_next, model[i + 1].da, model[i + 1].units,
                                                                    m);
            g_prime_z = d_g(model->activation_function, model[i].z, model[i].units, m);
            a_prev_t = transpose(model[i - 1].a, model[i - 1].units, m);
            da = matrix_multiplication(w_next_t, g_prime_z_next_ElementWiseMul_a_next, model[i].units,
                                       model[i + 1].units, m);
            db = element_wise_mul(da, g_prime_z, model[i].units, m);
            dw = matrix_multiplication(db, a_prev_t, model[i].units, m, model[i - 1].units);
            dw = matrix_constant_mul(dw, constant, model[i].units, model[i - 1].units);
            db = sum_axis(db, 1, model[i].units, m);
            db = matrix_constant_mul(db, constant, model[i].units, 1);
            model[i].da = da;
            model[i].dw = dw;
            model[i].db = db;
        }
    }

    w_next_t = transpose(model[1].w, model[1].units, model[0].units);
    g_prime_z_next = d_g(model->activation_function, model[1].z, model[1].units, model[0].units);
    g_prime_z_next_ElementWiseMul_a_next = element_wise_mul(g_prime_z_next, model[1].da, model[1].units, m);
    g_prime_z = d_g(model->activation_function, model[0].z, model[0].units, m);
    a_prev_t = transpose(inputs, input_number, m);
    da = matrix_multiplication(w_next_t, g_prime_z_next_ElementWiseMul_a_next, model[0].units, model[1].units, m);
    db = element_wise_mul(da, g_prime_z, model[0].units, m);
    dw = matrix_multiplication(db, a_prev_t, model[0].units, m, input_number);
    dw = matrix_constant_mul(dw, constant, model[0].units, input_number);
    db = sum_axis(db, 1, model[0].units, m);
    db = matrix_constant_mul(db, constant, model[0].units, 1);
    model[0].da = da;
    model[0].dw = dw;
    model[0].db = db;
}


#endif //C_CODE_DNN_H
