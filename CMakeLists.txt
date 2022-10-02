cmake_minimum_required(VERSION 3.21)
project(C_code C)

set(CMAKE_C_STANDARD 99)

add_executable(C_code main.c Matrix.h Activation_functions.h DNN.h Loss_functions.h Optimizers.h)
