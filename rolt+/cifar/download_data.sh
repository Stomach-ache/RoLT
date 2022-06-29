#!/bin/bash

dataset=$1

if [ $dataset == "cifar10" ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz --strip-components 1
    rm cifar-10-python.tar.gz
elif [ $dataset == "cifar100" ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    tar -xzvf cifar-100-python.tar.gz --strip-components 1
    rm cifar-100-python.tar.gz
fi