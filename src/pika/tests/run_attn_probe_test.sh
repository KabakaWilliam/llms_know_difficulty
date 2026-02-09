#!/bin/bash
# Run the sklearn probe test from the tests directory
set -e

# Get the project src directory (parent of parent of tests)
SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

# Run the test module
cd "$SRC_DIR"
export CUDA_VISIBLE_DEVICES=0
python -m pika.tests.test_attn_probe_real_data "$@"
# python -m pika.tests.test_sklearn_probe_e2e "$@"