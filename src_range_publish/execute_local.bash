#!/bin/bash

conda activate FREEDOM
export CUDA_VISIBLE_DEVICES=$2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

$1