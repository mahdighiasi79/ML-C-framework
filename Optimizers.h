//
// Created by mmghf on 10/1/2022.
//

#ifndef C_CODE_OPTIMIZERS_H
#define C_CODE_OPTIMIZERS_H


#include <math.h>
#include <string.h>

#include "dataTypes.h"
#include "Matrix.h"
#include "Loss_functions.h"
#include "DNN.h"


void sgd_optimizer_step(Optimizer_input sgdInput) {

    Layer *model = sgdInput.model;
    int layers = sgdInput.layers;
    char *loss_function = sgdInput.loss_function;
    double **inputs = sgdInput.inputs;
    double **labels = sgdInput.labels;
    int input_number = sgdInput.input_number;
    int m = sgdInput.m;
    double learning_rate = sgdInput.learning_rate;

    forward_prop(model, layers, inputs, input_number, m);
    back_prop(model, layers, loss_function, inputs, labels, input_number, m);

    double **lr_dw = matrix_constant_mul(model[0].dw, learning_rate, model[0].units, input_number);
    model[0].w = matrix_subtraction(model[0].w, lr_dw, model[0].units, input_number);
    double **lr_db = matrix_constant_mul(model[0].db, learning_rate, model[0].units, 1);
    model[0].b = matrix_subtraction(model[0].b, lr_db, model[0].units, 1);
    for (int i = 1; i < layers; i++) {
        lr_dw = matrix_constant_mul(model[i].dw, learning_rate, model[i].units, model[i - 1].units);
        model[i].w = matrix_subtraction(model[i].w, lr_dw, model[i].units, model[i - 1].units);
        lr_db = matrix_constant_mul(model[i].db, learning_rate, model[i].units, 1);
        model[i].b = matrix_subtraction(model[i].b, lr_db, model[i].units, 1);
    }
}


void momentum_sgd_step(Optimizer_input optimizerInput) {

    Layer *model = optimizerInput.model;
    int layers = optimizerInput.layers;
    char *loss_function = optimizerInput.loss_function;
    double **inputs = optimizerInput.inputs;
    double **labels = optimizerInput.labels;
    int input_number = optimizerInput.input_number;
    int m = optimizerInput.m;
    double learning_rate = optimizerInput.learning_rate;
    double beta = optimizerInput.beta;

    forward_prop(model, layers, inputs, input_number, m);
    back_prop(model, layers, loss_function, inputs, labels, input_number, m);

    double **bVdw = matrix_constant_mul(model[0].Vdw, beta, model[0].units, input_number);
    double **bdw = matrix_constant_mul(model[0].dw, 1 - beta, model[0].units, input_number);
    double **Vdw = matrix_addition(bVdw, bdw, model[0].units, input_number);
    double **bVdb = matrix_constant_mul(model[0].Vdb, beta, model[0].units, 1);
    double **bdb = matrix_constant_mul(model[0].db, 1 - beta, model[0].units, 1);
    double **Vdb = matrix_addition(bVdb, bdb, model[0].units, 1);
    double **lr_Vdw = matrix_constant_mul(Vdw, learning_rate, model[0].units, input_number);
    model[0].w = matrix_subtraction(model[0].w, lr_Vdw, model[0].units, input_number);
    double **lr_Vdb = matrix_constant_mul(Vdb, learning_rate, model[0].units, 1);
    model[0].b = matrix_subtraction(model[0].b, lr_Vdb, model[0].units, 1);
    model[0].Vdw = Vdw;
    model[0].Vdb = Vdb;
    for (int i = 1; i < layers; i++) {
        bVdw = matrix_constant_mul(model[i].Vdw, beta, model[i].units, model[i - 1].units);
        bdw = matrix_constant_mul(model[i].dw, 1 - beta, model[i].units, model[i - 1].units);
        Vdw = matrix_addition(bVdw, bdw, model[i].units, model[i - 1].units);
        bVdb = matrix_constant_mul(model[i].Vdb, beta, model[i].units, 1);
        bdb = matrix_constant_mul(model[i].db, 1 - beta, model[i].units, 1);
        Vdb = matrix_addition(bVdb, bdb, model[i].units, 1);
        lr_Vdw = matrix_constant_mul(Vdw, learning_rate, model[i].units, model[i - 1].units);
        model[i].w = matrix_subtraction(model[i].w, lr_Vdw, model[i].units, model[i - 1].units);
        lr_Vdb = matrix_constant_mul(Vdb, learning_rate, model[i].units, 1);
        model[i].b = matrix_subtraction(model[i].b, lr_Vdb, model[i].units, 1);
        model[i].Vdw = Vdw;
        model[i].Vdb = Vdb;
    }
}


void RMS_prop_step(Optimizer_input optimizerInput) {

    Layer *model = optimizerInput.model;
    int layers = optimizerInput.layers;
    char *loss_function = optimizerInput.loss_function;
    double **inputs = optimizerInput.inputs;
    double **labels = optimizerInput.labels;
    int input_number = optimizerInput.input_number;
    int m = optimizerInput.m;
    double learning_rate = optimizerInput.learning_rate;
    double beta = optimizerInput.beta;
    double epsilon = optimizerInput.epsilon;

    forward_prop(model, layers, inputs, input_number, m);
    back_prop(model, layers, loss_function, inputs, labels, input_number, m);

    double **bSdw = matrix_constant_mul(model[0].Sdw, beta, model[0].units, input_number);
    double **dw_2 = power_matrix(model[0].dw, 2, model[0].units, input_number);
    double **bdw_2 = matrix_constant_mul(dw_2, 1 - beta, model[0].units, input_number);
    double **Sdw = matrix_addition(bSdw, bdw_2, model[0].units, input_number);
    double **bSdb = matrix_constant_mul(model[0].Sdb, beta, model[0].units, 1);
    double **db_2 = power_matrix(model[0].db, 2, model[0].units, input_number);
    double **bdb_2 = matrix_constant_mul(db_2, 1 - beta, model[0].units, 1);
    double **Sdb = matrix_addition(bSdb, bdb_2, model[0].units, 1);
    double **update = power_matrix(Sdw, 0.5, model[0].units, input_number);
    update = matrix_constant_add(update, epsilon, model[0].units, input_number);
    update = element_wise_div(model[0].dw, update, model[0].units, input_number);
    update = matrix_constant_mul(update, learning_rate, model[0].units, input_number);
    model[0].w = matrix_subtraction(model[0].w, update, model[0].units, input_number);
    update = power_matrix(Sdb, 0.5, model[0].units, input_number);
    update = matrix_constant_add(update, epsilon, model[0].units, input_number);
    update = element_wise_div(model[0].dw, update, model[0].units, input_number);
    update = matrix_constant_mul(update, learning_rate, model[0].units, input_number);
    model[0].b = matrix_subtraction(model[0].b, update, model[0].units, input_number);
    model[0].Sdw = Sdw;
    model[0].Sdb = Sdb;
    for (int i = 1; i < layers; i++) {
        bSdw = matrix_constant_mul(model[i].Sdw, beta, model[i].units, model[i - 1].units);
        dw_2 = power_matrix(model[i].dw, 2, model[i].units, model[i - 1].units);
        bdw_2 = matrix_constant_mul(dw_2, 1 - beta, model[i].units, model[i - 1].units);
        Sdw = matrix_addition(bSdw, bdw_2, model[i].units, model[i - 1].units);
        bSdb = matrix_constant_mul(model[i].Sdb, beta, model[i].units, 1);
        db_2 = power_matrix(model[0].db, 2, model[i].units, model[i - 1].units);
        bdb_2 = matrix_constant_mul(db_2, 1 - beta, model[i].units, 1);
        Sdb = matrix_addition(bSdb, bdb_2, model[i].units, 1);
        update = power_matrix(Sdw, 0.5, model[i].units, model[i - 1].units);
        update = matrix_constant_add(update, epsilon, model[i].units, model[i - 1].units);
        update = element_wise_div(model[i].dw, update, model[i].units, model[i - 1].units);
        update = matrix_constant_mul(update, learning_rate, model[i].units, model[i - 1].units);
        model[i].w = matrix_subtraction(model[i].w, update, model[i].units, model[i - 1].units);
        update = power_matrix(Sdb, 0.5, model[i].units, model[i - 1].units);
        update = matrix_constant_add(update, epsilon, model[i].units, model[i - 1].units);
        update = element_wise_div(model[i].dw, update, model[i].units, model[i - 1].units);
        update = matrix_constant_mul(update, learning_rate, model[i].units, model[i - 1].units);
        model[i].b = matrix_subtraction(model[i].b, update, model[i].units, model[i - 1].units);
        model[i].Sdw = Sdw;
        model[i].Sdb = Sdb;
    }
}


void Adam(Optimizer_input optimizerInput) {

    Layer *model = optimizerInput.model;
    int layers = optimizerInput.layers;
    char *loss_function = optimizerInput.loss_function;
    double **inputs = optimizerInput.inputs;
    double **labels = optimizerInput.labels;
    int input_number = optimizerInput.input_number;
    int m = optimizerInput.m;
    double learning_rate = optimizerInput.learning_rate;
    double beta = optimizerInput.beta;
    double alpha = optimizerInput.alpha;
    double epsilon = optimizerInput.epsilon;
    int *t = optimizerInput.iteration;


    forward_prop(model, layers, inputs, input_number, m);
    back_prop(model, layers, loss_function, inputs, labels, input_number, m);


    double **bVdw = matrix_constant_mul(model[0].Vdw, beta, model[0].units, input_number);
    double **bdw = matrix_constant_mul(model[0].dw, 1 - beta, model[0].units, input_number);
    double **Vdw = matrix_addition(bVdw, bdw, model[0].units, input_number);
    double **Vdw_corrected = matrix_constant_mul(Vdw, 1 / (1 - pow(beta, *t)), model[0].units, input_number);
    double **bVdb = matrix_constant_mul(model[0].Vdb, beta, model[0].units, 1);
    double **bdb = matrix_constant_mul(model[0].db, 1 - beta, model[0].units, 1);
    double **Vdb = matrix_addition(bVdb, bdb, model[0].units, 1);
    double **Vdb_corrected = matrix_constant_mul(Vdb, 1 / (1 - pow(beta, *t)), model[0].units, 1);

    double **bSdw = matrix_constant_mul(model[0].Sdw, alpha, model[0].units, input_number);
    double **dw_2 = power_matrix(model[0].dw, 2, model[0].units, input_number);
    double **bdw_2 = matrix_constant_mul(dw_2, 1 - alpha, model[0].units, input_number);
    double **Sdw = matrix_addition(bSdw, bdw_2, model[0].units, input_number);
    double **Sdw_corrected = matrix_constant_mul(Sdw, 1 / (1 - pow(alpha, *t)), model[0].units, input_number);
    double **bSdb = matrix_constant_mul(model[0].Sdb, alpha, model[0].units, 1);
    double **db_2 = power_matrix(model[0].db, 2, model[0].units, input_number);
    double **bdb_2 = matrix_constant_mul(db_2, 1 - alpha, model[0].units, 1);
    double **Sdb = matrix_addition(bSdb, bdb_2, model[0].units, 1);
    double **Sdb_corrected = matrix_constant_mul(Sdb, 1 / (1 - pow(alpha, *t)), model[0].units, 1);

    double **update = power_matrix(Sdw_corrected, 0.5, model[0].units, input_number);
    update = matrix_constant_add(update, epsilon, model[0].units, input_number);
    update = element_wise_div(Vdw_corrected, update, model[0].units, input_number);
    update = matrix_constant_mul(update, learning_rate, model[0].units, input_number);
    model[0].w = matrix_subtraction(model[0].w, update, model[0].units, input_number);
    model[0].Sdw = Sdw;

    update = power_matrix(Sdb_corrected, 0.5, model[0].units, 1);
    update = matrix_constant_add(update, epsilon, model[0].units, 1);
    update = element_wise_div(Vdb_corrected, update, model[0].units, 1);
    update = matrix_constant_mul(update, learning_rate, model[0].units, 1);
    model[0].b = matrix_subtraction(model[0].b, update, model[0].units, 1);
    model[0].Sdb = Sdb;

    for (int i = 1; i < layers; i++) {

        bVdw = matrix_constant_mul(model[i].Vdw, beta, model[i].units, model[i - 1].units);
        bdw = matrix_constant_mul(model[i].dw, 1 - beta, model[i].units, model[i - 1].units);
        Vdw = matrix_addition(bVdw, bdw, model[i].units, model[i - 1].units);
        Vdw_corrected = matrix_constant_mul(Vdw, 1 / (1 - pow(beta, *t)), model[i].units, model[i - 1].units);
        bVdb = matrix_constant_mul(model[i].Vdb, beta, model[i].units, 1);
        bdb = matrix_constant_mul(model[i].db, 1 - beta, model[i].units, 1);
        Vdb = matrix_addition(bVdb, bdb, model[i].units, 1);
        Vdb_corrected = matrix_constant_mul(Vdb, 1 / (1 - pow(beta, *t)), model[i].units, 1);

        bSdw = matrix_constant_mul(model[i].Sdw, alpha, model[i].units, model[i - 1].units);
        dw_2 = power_matrix(model[i].dw, 2, model[i].units, model[i - 1].units);
        bdw_2 = matrix_constant_mul(dw_2, 1 - alpha, model[i].units, model[i - 1].units);
        Sdw = matrix_addition(bSdw, bdw_2, model[i].units, model[i - 1].units);
        Sdw_corrected = matrix_constant_mul(Sdw, 1 / (1 - pow(alpha, *t)), model[i].units, model[i - 1].units);
        bSdb = matrix_constant_mul(model[i].Sdb, alpha, model[i].units, 1);
        db_2 = power_matrix(model[i].db, 2, model[i].units, model[i - 1].units);
        bdb_2 = matrix_constant_mul(db_2, 1 - alpha, model[i].units, 1);
        Sdb = matrix_addition(bSdb, bdb_2, model[i].units, 1);
        Sdb_corrected = matrix_constant_mul(Sdb, 1 / (1 - pow(alpha, *t)), model[i].units, 1);

        update = power_matrix(Sdw_corrected, 0.5, model[i].units, model[i - 1].units);
        update = matrix_constant_add(update, epsilon, model[i].units, model[i - 1].units);
        update = element_wise_div(Vdw_corrected, update, model[i].units, model[i - 1].units);
        update = matrix_constant_mul(update, learning_rate, model[i].units, model[i - 1].units);
        model[i].w = matrix_subtraction(model[i].w, update, model[i].units, model[i - 1].units);
        model[i].Sdw = Sdw;

        update = power_matrix(Sdb_corrected, 0.5, model[i].units, 1);
        update = matrix_constant_add(update, epsilon, model[i].units, 1);
        update = element_wise_div(Vdb_corrected, update, model[i].units, 1);
        update = matrix_constant_mul(update, learning_rate, model[i].units, 1);
        model[i].b = matrix_subtraction(model[i].b, update, model[i].units, 1);
        model[i].Sdb = Sdb;
    }


    (*optimizerInput.iteration)++;
}


void optimizer_step(Optimizer_input optimizerInput) {
    char *optimizer = optimizerInput.optimizer;
    if (strcmp(optimizer, "sgd") == 0)
        sgd_optimizer_step(optimizerInput);
    else if (strcmp(optimizer, "momentum_sgd") == 0)
        momentum_sgd_step(optimizerInput);
    else if (strcmp(optimizer, "RMS_prop") == 0)
        RMS_prop_step(optimizerInput);
    else if (strcmp(optimizer, "Adam") == 0) {
        Adam(optimizerInput);
    }
    printf("not a valid optimizer!\n");
}


void Optimize(Optimizer_input optimizerInput) {

    Layer *model = optimizerInput.model;
    int layers = optimizerInput.layers;
    char *loss_function = optimizerInput.loss_function;
    double **inputs = optimizerInput.inputs;
    double **labels = optimizerInput.labels;
    int m = optimizerInput.m;
    int batch_size = optimizerInput.batch_size;
    int epochs = optimizerInput.epochs;
    int input_number = optimizerInput.input_number;
    (*optimizerInput.iteration) = 0;

    int number_of_batches = floor((double) m / batch_size);

    for (int i = 0; i < epochs; i++) {

        inputs = shuffle(inputs, m);

        for (int j = 0; j < number_of_batches; j++) {
            double **x = copy_matrix(inputs, j * batch_size, (j + 1) * batch_size, input_number);
            double **y = copy_matrix(labels, j * batch_size, (j + 1) * batch_size, model[layers - 1].units);
            x = transpose(x, batch_size, input_number);
            y = transpose(y, batch_size, model[layers - 1].units);
            optimizerInput.inputs = x;
            optimizerInput.labels = y;
            optimizerInput.m = batch_size;
            optimizer_step(optimizerInput);
            printf("loss: %f\n", loss(loss_function, model[layers - 1].a, y, model[layers - 1].units, batch_size));
        }
    }
}


#endif //C_CODE_OPTIMIZERS_H
