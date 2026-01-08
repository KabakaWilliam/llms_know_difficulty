#!/bin/bash
# Run the sklearn probe test from the tests directory
set -e

# Get the project src directory (parent of parent of tests)
SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

# Run the test module
cd "$SRC_DIR"
python -m llms_know_difficulty.test_sklearn_probe_e2e "$@"
