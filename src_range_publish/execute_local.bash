#!/bin/bash

conda activate FREEDOM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=$2

$1