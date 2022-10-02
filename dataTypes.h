//
// Created by mmghf on 9/30/2022.
//

#ifndef C_DATATYPES_H
#define C_DATATYPES_H


typedef struct layer {
    char *type;
    char *activation_function;
    int units;
    int number_of_filters;
    int filter;
    int padding;
    int stride;
    double **w;
    double **b;
    double **z;
    double **a;
    double **dw;
    double **db;
    double **dz;
    double **da;
    double **Vdw;
    double **Sdw;
    double **Vdb;
    double **Sdb;
    double ****wc;
    double ****bc;
    double ****zc;
    double ****ac;
    double ****dwc;
    double ****dbc;
    double ****dzc;
    double ****dac;
    double ****Vdwc;
    double ****Vdbc;
    double ****Sdwc;
    double ****Sdbc;
} Layer;


typedef struct optimizer_input {
    char *optimizer;
    Layer *model;
    int layers;
    char *loss_function;
    double **inputs;
    double **labels;
    int input_number;
    int m;
    int batch_size;
    int epochs;
    double learning_rate;
    double beta;
    double alpha;
    double epsilon;
    int *iteration;
} Optimizer_input;


typedef struct cache {
    double **a1;
    double **a2;
    double **a3;
    double **z1;
    double **z2;
    double **z3;
} Cache;


typedef struct grad {
    double **dw1;
    double **dw2;
    double **dw3;
    double **db1;
    double **db2;
    double **db3;
} Grads;


#endif //C_DATATYPES_H
