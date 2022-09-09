#include <stdio.h>
#include <stdlib.h>

#include "DNN.h"

#define magic_number_size 4


int number_of_train_images;
int number_of_test_images;
int rows;
int columns;
int *train_labels;
int *test_labels;
int **train_images;
int **test_images;


int readInteger(FILE *fp) {

    unsigned char buffer[4];
    fread(buffer, 1, 4, fp);

    int result = 0;
    for (int i = 0; i < 4; i++) {
        result *= 256;
        result += buffer[i];
    }

    return result;
}


int **readImages(char *address, int t_s) {

    FILE *fp = fopen(address, "rb");
    fseek(fp, magic_number_size, SEEK_SET);

    int images = readInteger(fp);
    rows = readInteger(fp);
    columns = readInteger(fp);
    int pixels = rows * columns;

    int **data = (int **) malloc(images * sizeof(int *));
    for (int i = 0; i < images; i++)
        data[i] = (int *) malloc(pixels * sizeof(int));

    unsigned char pixel;
    for (int i = 0; i < images; i++) {

        for (int j = 0; j < pixels; j++) {

            fread(&pixel, 1, 1, fp);
            data[i][j] = pixel;
        }
    }

    // variable t_s indicates whether this function is reading train data or test data
    if (t_s == 1)
        number_of_train_images = images;
    else
        number_of_test_images = images;

    return data;
}


int *readLabels(char *address) {

    FILE *fp = fopen(address, "rb");
    fseek(fp, magic_number_size, SEEK_SET);

    int items = readInteger(fp);
    int *labels = (int *) malloc(items * sizeof(int));
    unsigned char label;
    for (int i = 0; i < items; i++) {
        fread(&label, 1, 1, fp);
        labels[i] = label;
    }

    return labels;
}


void getDataset() {

    train_images = readImages("D:\\computer\\ComputationalIntelligence\\NeuralNetwork\\p1\\train-images.idx3-ubyte", 1);
    test_images = readImages("D:\\computer\\ComputationalIntelligence\\NeuralNetwork\\p1\\t10k-images.idx3-ubyte", 0);
    train_labels = readLabels("D:\\computer\\ComputationalIntelligence\\NeuralNetwork\\p1\\train-labels.idx1-ubyte");
    test_labels = readLabels("D:\\computer\\ComputationalIntelligence\\NeuralNetwork\\p1\\t10k-labels.idx1-ubyte");

    for (int i = 0; i < number_of_train_images; i++) {
        for (int j = 0; j < rows * columns; j++)
            train_images[i][j] /= 256;
    }

    for (int i = 0; i < number_of_test_images; i++) {
        for (int j = 0; j < rows * columns; j++)
            test_images[i][j] /= 256;
    }
}


int main() {
    getDataset();
    int number_of_layers = 4;
    int layers_sizes[] = {784, 16, 16, 10};
    initialize_ANN(number_of_layers, layers_sizes, "sigmoid");
    return 0;
}
