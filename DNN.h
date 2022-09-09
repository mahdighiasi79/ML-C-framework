//
// Created by Mahdi on 5/2/2022.
//

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>

#include "Matrix.h"
#include "Activation_functions.h"

#ifndef C_CODE_DNN_H
#define C_CODE_DNN_H

typedef struct ann {

    int number_of_layers;
    int *layers_sizes;

    double ***w;
    double ***b;

    double ***layers;

    double ***z;

    char *activation_function;

} ANN;

double standard_normal_distribution(int input) {

    double x = (double) input;
    double PI = acos(-1);

    double coefficient = 1 / pow(2 * PI, 0.5);
    double exponent = -pow(x, 2) / 2;
    double result = coefficient * exp(exponent);

    return result;
}


ANN *initialize_ANN(int num_layers, int *layers_sizes, char *activation_function) {

    time_t t = time(NULL);
    srand(t);

    ANN *ann = (ANN *) malloc(sizeof(ANN));
    ann->number_of_layers = num_layers;
    ann->layers_sizes = layers_sizes;
    ann->activation_function = activation_function;


    ann->w = (double ***) malloc((num_layers - 1) * sizeof(double **));

    for (int i = 0; i < num_layers - 1; i++) {
        ann->w[i] = (double **) malloc(layers_sizes[i + 1] * sizeof(double *));

        for (int j = 0; j < layers_sizes[i + 1]; j++) {
            ann->w[i][j] = (double *) malloc(layers_sizes[i] * sizeof(double));

            for (int k = 0; k < layers_sizes[i]; k++)
                ann->w[i][j][k] = standard_normal_distribution(rand());
        }
    }


    ann->b = (double ***) malloc((num_layers - 1) * sizeof(double **));

    for (int i = 0; i < num_layers - 1; i++) {
        int neurons = layers_sizes[i + 1];
        ann->b[i] = (double **) malloc(neurons * sizeof(double *));

        for (int j = 0; j < neurons; j++) {
            ann->b[i][j] = (double *) malloc(sizeof(double));
            ann->b[i][j][0] = 0;
        }
    }


    ann->layers = (double ***) malloc(num_layers * sizeof(double **));
    ann->layers[0] = (double **) malloc(layers_sizes[0] * sizeof(double *));

    ann->z = (double ***) malloc((num_layers - 1) * sizeof(double **));

    return ann;
}


double *feed_forward(const double *input, ANN *ann) {

    for (int i = 0; i < ann->layers_sizes[0]; i++)
        ann->layers[0][i][0] = input[i];

    for (int i = 0; i < ann->number_of_layers - 1; i++) {

        double **weights = ann->w[i];
        double **layer = ann->layers[i];
        double **biases = ann->b[i];
        int row = ann->layers_sizes[i + 1];
        int column = ann->layers_sizes[i];

        double **hidden_layer = matrix_multiplication(weights, layer, row, column, 1);
        hidden_layer = matrix_addition(hidden_layer, biases, row, 1);
        ann->z[i] = hidden_layer;

#pragma omp parallel for
        for (int j = 0; j < row; j++)
            hidden_layer[j][0] = call(hidden_layer[j][0], ann->activation_function);

        ann->layers[i + 1] = hidden_layer;
    }

    int output_layer_size = ann->layers_sizes[ann->number_of_layers - 1];
    double *result = (double *) malloc( output_layer_size * sizeof(double));
    for (int i = 0; i < output_layer_size; i++)
        result[i] = ann->layers[ann->number_of_layers - 1][i][0];

    return result;
}

double weight_back_propagation(ANN ann, const int *weight_position, const int *neuron_position, double derivation) {

    int weight_layer = weight_position[0];
    int weight_row = weight_position[1];
    int weight_column = weight_position[2];

    int neuron_layer = neuron_position[0];
    int neuron_index = neuron_position[1];

    if (weight_layer == neuron_layer) {

        if (neuron_index != weight_row)
            return 0;

        double z = ann.z[neuron_layer - 2][neuron_index - 1][0];
        double neuron = ann.layers[neuron_layer - 2][weight_column - 1][0];
        return d_call(z, ann.activation_function) * neuron * derivation;
    }


    int next_layer = neuron_layer - 1;
    int next_layer_size = ann.layers_sizes[next_layer - 1];
    double z = ann.z[neuron_layer - 2][neuron_index - 1][0];
    double s = d_call(z, ann.activation_function);
    double *weights = ann.w[neuron_layer - 2][neuron_index - 1];
    derivation *= s;
    double derivation_copy = derivation;
    int *newNeuron_position = (int *) malloc(2 * sizeof(int));
    newNeuron_position[0] = neuron_layer - 1;

    for (int i = 0; i < next_layer_size; i++) {
        double w = weights[i];
        newNeuron_position[1] = i + 1;
        derivation += w * weight_back_propagation(ann, weight_position, newNeuron_position, derivation_copy);
    }

    return derivation;
}


double bias_back_propagation(ANN ann, const int *destination_neuron, const int *current_neuron, double derivation) {

    if ((destination_neuron[0] == current_neuron[0]) && (destination_neuron[1] == current_neuron[1])) {
        derivation *= d_call(ann.z[current_neuron[0] - 2][current_neuron[1] - 1][0], "sigmoid");
        return derivation;
    }


    int neuron_layer = current_neuron[0];
    int neuron_index = current_neuron[1];

    int next_layer = neuron_layer - 1;
    int next_layer_size = ann.layers_sizes[next_layer - 1];
    double z = ann.z[neuron_layer - 2][neuron_index - 1][0];
    double s = d_call(z, ann.activation_function);
    double *weights = ann.w[neuron_layer - 2][neuron_index - 1];
    double derivation_copy = derivation;
    int *newNeuron_position = (int *) malloc(2 * sizeof(int));
    newNeuron_position[0] = neuron_layer - 1;

    for (int i = 0; i < next_layer_size; i++) {
        double w = weights[i];
        newNeuron_position[1] = i + 1;
        derivation += s * w * bias_back_propagation(ann, destination_neuron, newNeuron_position, derivation_copy);
    }

    return derivation;
}


double Derivation(ANN ann, int *position, bool w_b, const double *train_labels) {

    int output_layer_size = ann.layers_sizes[ann.number_of_layers - 1];
    double **output_layer = ann.layers[ann.number_of_layers - 1];
    int neuron_position[2];
    neuron_position[0] = ann.number_of_layers;
    double derivation = 0;

    for (int i = 0; i < output_layer_size; i++) {
        double d = 2 * (output_layer[i][0] - train_labels[i]);
        neuron_position[1] = i + 1;
        if (w_b)
            d *= weight_back_propagation(ann, position, neuron_position, 1);
        else
            d *= bias_back_propagation(ann, position, neuron_position, 1);
        derivation += d;
    }

    return derivation;
}

double ****Gradient_matrix(ANN ann, double *train_labels) {

    double ****gradient_matrix = (double ****) malloc(2 * sizeof(double ***));


    double ***weights = (double ***) malloc((ann.number_of_layers - 1) * sizeof(double **));

    for (int i = 0; i < ann.number_of_layers - 1; i++) {

        int current_layer_size = ann.layers_sizes[i];
        int next_layer_size = ann.layers_sizes[i + 1];

        weights[i] = (double **) malloc(next_layer_size * sizeof(double *));

        for (int j = 0; j < next_layer_size; j++) {

            weights[i][j] = (double *) malloc(current_layer_size * sizeof(double));

            for (int k = 0; k < current_layer_size; k++) {
                int position[3];
                position[0] = i + 2;
                position[1] = j + 1;
                position[2] = k + 1;
                weights[i][j][k] = Derivation(ann, position, true, train_labels);
            }
        }
    }
    gradient_matrix[0] = weights;


    double ***biases = (double ***) malloc((ann.number_of_layers - 1) * sizeof(double **));

    for (int i = 0; i < ann.number_of_layers - 1; i++) {

        int current_layer_size = ann.layers_sizes[i + 1];
        biases[i] = (double **) malloc(current_layer_size * sizeof(double *));

        for (int j = 0; j < current_layer_size; j++) {
            biases[i][j] = (double *) malloc(sizeof(double));
            int position[2];
            position[0] = i + 2;
            position[1] = j + 1;
            biases[i][j][0] = Derivation(ann, position, false, train_labels);
        }
    }
    gradient_matrix[1] = biases;


    return gradient_matrix;
}

#endif //C_CODE_DNN_H
