#!/bin/bash -v

make clean && make -j4

# ./bin/sgemm_nn_naive
# ./bin/sgemm_nn_smem
./bin/sgemm_nn_reg