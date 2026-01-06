# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted to use Sandbox Fusion for secure code execution
# Original borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py

import concurrent.futures
import json
import logging
import os
import threading
import traceback
from typing import Any, Optional

# Import sandbox fusion utilities
from ..code_utils import call_sandbox_api

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
SANDBOX_FUSION_URL = os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080/run_code")


def _process_single_test_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    generation: str,
    timeout: int,
    memory_limit_mb: int = 1024,
    language: str = "python",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    fn_name: Optional[str] = None,
) -> tuple[int, dict[str, Any]]:
    """Process a single test case using sandbox fusion."""
    api_response = None
    error_msg = None

    current_generation_code = generation

    if fn_name and language == "python":
        # Wrapper for function-based problems (with JSON input parsing)
        wrapper_code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: "
                              f"{{_raw_input_str[:200]}}\\n")
            return None, True # result, error_occurred

    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method
            _solution_instance = _Solution_class()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred

        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        # Serialize result to stdout
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result)) # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
"""
        current_generation_code = wrapper_code

    stdin = None if stdin_data is None else str(stdin_data)

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_sandbox_api(
                    sandbox_fusion_url=SANDBOX_FUSION_URL,
                    code=current_generation_code,
                    stdin=stdin,
                    compile_timeout=timeout,
                    run_timeout=timeout,
                    memory_limit_mb=memory_limit_mb,
                    language=language,
                )
        else:
            api_response, error_msg = call_sandbox_api(
                sandbox_fusion_url=SANDBOX_FUSION_URL,
                code=current_generation_code,
                stdin=stdin,
                compile_timeout=timeout,
                run_timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                language=language,
            )
    except Exception as e:
        error_msg = f"Sandbox API Exception: {e}"
        logger.error(f"Case {case_index}: {error_msg}")
        traceback.print_exc()

    metadata = {
        "case_index": case_index,
        "input": stdin,
        "expected_output": str(expected_output) if expected_output else None,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
    }

    result_status = -1  # Default error

    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1
        logger.error(f"Case {case_index}: API error: {error_msg}")
        print(f"Case {case_index}: API error: {error_msg}")  # Also print to console
    elif api_response is not None:
        metadata["api_response"] = api_response
        api_status = api_response.get("status")
        compile_result = api_response.get("compile_result") or {}
        run_result = api_response.get("run_result") or {}

        metadata["api_status"] = api_status
        metadata["compile_status"] = compile_result.get("status")
        metadata["run_status"] = run_result.get("status")
        metadata["stdout"] = run_result.get("stdout")
        metadata["stderr"] = run_result.get("stderr")
        metadata["exit_code"] = run_result.get("return_code")
        metadata["duration"] = run_result.get("execution_time")

        # Determine result status
        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
            result_status = -1
        elif api_status == "Failed":
            # Check for compile error
            if compile_result.get("status") in ["Error", "TimeLimitExceeded"]:
                metadata["status"] = "compile_error"
                result_status = -4
            # Check for run timeout
            elif run_result.get("status") == "TimeLimitExceeded":
                metadata["status"] = "timeout"
                result_status = -3
            # Check for runtime error
            elif run_result.get("return_code") != 0:
                metadata["status"] = "runtime_error"
                result_status = -2
            else:
                metadata["status"] = "unknown_error"
                result_status = -1
        elif api_status == "Success":
            # Run completed successfully, check if output matches
            if run_result and run_result.get("status") == "Finished":
                actual_output = run_result.get("stdout", "").strip()
                expected_str = str(expected_output).strip()
                
                # Try exact match first
                if actual_output == expected_str:
                    metadata["status"] = "passed"
                    result_status = True
                else:
                    # Try JSON comparison if both are valid JSON
                    try:
                        actual_json = json.loads(actual_output)
                        expected_json = json.loads(expected_str)
                        if actual_json == expected_json:
                            metadata["status"] = "passed"
                            result_status = True
                        else:
                            metadata["status"] = "wrong_answer"
                            result_status = False
                    except (json.JSONDecodeError, TypeError):
                        # Fall back to string comparison
                        metadata["status"] = "wrong_answer"
                        result_status = False
            else:
                metadata["status"] = "unknown_error"
                result_status = -1
        else:
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1
    else:
        # Both api_response and error_msg are None - shouldn't happen but handle it
        metadata["status"] = "unknown_api_state"
        result_status = -1
        logger.error(f"Case {case_index}: Both api_response and error_msg are None")
        print(f"Case {case_index}: Both api_response and error_msg are None")

    return result_status, metadata


def check_correctness(
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    debug: bool = True,
    memory_limit_mb: int = 1024,
    language: str = "python",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Check correctness of code generation using Sandbox Fusion.
    
    Args:
        in_outs: Dictionary containing "inputs" and "outputs" lists
        generation: The generated code string (may contain markdown)
        timeout: Timeout for each test case
        debug: Enable debug logging
        memory_limit_mb: Memory limit for sandbox execution
        language: Programming language ("python", "cpp", etc.)
        concurrent_semaphore: Optional semaphore for concurrent execution control
    
    Returns:
        A tuple (results, metadata_list) where:
        - results: List of booleans/error codes for each test case
        - metadata_list: List of metadata dicts for each test case
    """
    
    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]

    # Extract code from markdown if present (same as compute_score does)
    solution = generation
    if "```python" in generation:
        solution = generation.split("```python")[-1].split("```")[0]
    elif "```" in generation:
        # Handle cases like ```\ncode\n```
        parts = generation.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    solution = rest

    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)
    
    if num_cases == 0:
        logger.warning("Empty inputs provided.")
        return [], []

    if len(inputs) != len(expected_outputs):
        logger.warning(f"Input/output count mismatch: {len(inputs)} vs {len(expected_outputs)}")
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    results = [None] * num_cases
    metadata_list = [None] * num_cases

    # Process all test cases using ThreadPoolExecutor with limited concurrency
    # Limit to fewer workers to avoid overwhelming the sandbox
    max_workers = min(2, num_cases)  # Use at most 2 concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _process_single_test_case,
                i,
                inputs[i],
                expected_outputs[i],
                solution,  # Use extracted solution instead of raw generation
                timeout,
                memory_limit_mb,
                language,
                concurrent_semaphore,
                fn_name,
            ): i
            for i in range(num_cases)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata
            except Exception as exc:
                logger.error(f"Test case {index} failed with exception: {exc}")
                import traceback as tb
                tb.print_exc()  # Print full traceback
                results[index] = -1
                metadata_list[index] = {"error": f"Exception: {exc}", "case_index": index}

    if debug:
        logger.info(f"Correctness check finished. Results: {results}")

    return results, metadata_list