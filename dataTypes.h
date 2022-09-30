//
// Created by mmghf on 9/30/2022.
//

#ifndef C_DATATYPES_H
#define C_DATATYPES_H


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
