#!/bin/bash -v

nvcc -o ./bin/matMul ./src/constant.cpp ./test/matMul.cpp ./src/util/debug.cpp ./src/util/data.cpp -I./include -lcublas -std=c++11