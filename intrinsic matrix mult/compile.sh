#!/bin/bash
gcc -O3 main.c -fopenmp -march=native -lm -o main
