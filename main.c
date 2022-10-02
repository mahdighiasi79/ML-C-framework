#include "DNN.h"
#include "Optimizers.h"

int main() {

    // making a made up train data with all zeros
    int input_samples = 60000;
    int input_size = 784;
    int output_size = 10;
    double **inputs = (double **) malloc(input_samples * sizeof(double *));
    for (int i = 0; i < input_samples; i++)
        inputs[i] = (double *) malloc(input_size * sizeof(double));
    double **labels = (double **) malloc(input_samples * sizeof(double *));
    for (int i = 0; i < input_samples; i++)
        labels[i] = (double *) malloc(output_size * sizeof(double));

    // building an ANN model
    Layer model[3] = {Dense(16, "relu"), Dense(16, "relu"), Dense(10, "softMax")};
    initialize_model(model, 3, 784);

    // train the model with sgd, by the Sum of Squared Errors loss function
    Optimizer_input optimizerInput;
    optimizerInput.optimizer = "sgd";
    optimizerInput.model = model;
    optimizerInput.layers = 3;
    optimizerInput.loss_function = "SSE";
    optimizerInput.inputs = inputs;
    optimizerInput.labels = labels;
    optimizerInput.input_number = input_size;
    optimizerInput.m = input_samples;
    optimizerInput.batch_size = 50;
    optimizerInput.epochs = 5;
    optimizerInput.learning_rate = 0.01;
    Optimize(optimizerInput);

    // making a made up test data with all zeros
    input_samples = 10000;
    inputs = (double **) malloc(input_samples * sizeof(double *));
    for (int i = 0; i < input_samples; i++)
        inputs[i] = (double *) malloc(input_size * sizeof(double));
    labels = (double **) malloc(input_samples * sizeof(double *));
    for (int i = 0; i < input_samples; i++)
        labels[i] = (double *) malloc(output_size * sizeof(double));

    // getting the model's prediction
    forward_prop(model, 3, inputs, input_size, input_samples);
    double **predictions = model[2].a;
    // printing the predictions
    print_matrix(predictions, model[2].units, input_samples);
    return 0;
}
