# Offline verification script for code generations
# Verifies pre-generated code solutions against test cases without needing sandbox_fusion
# Can be run after generating solutions with create_code_generations_no_verification.py
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd
from tqdm import tqdm
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_offline(
    parquet_file: str,
    sandbox_fusion_url: str,
    output_parquet: Optional[str] = None,
    max_concurrent_requests: int = 8,
    memory_limit_mb: int = 1024,
    timeout: int = 10,
):
    """
    Offline verification of generated code solutions.
    
    Args:
        parquet_file: Path to the NO_VERIFY parquet file from create_code_generations_no_verification.py
        sandbox_fusion_url: Sandbox fusion API endpoint (e.g., "http://localhost:8080/run_code")
        output_parquet: Path to save verified results (default: replace NO_VERIFY with VERIFIED)
        max_concurrent_requests: Max concurrent sandbox API requests
        memory_limit_mb: Memory limit per execution
        timeout: Timeout for each test case
    """
    
    # Import here to avoid issues if sandbox utils not available
    from utils.code_utils import check_correctness
    import json
    
    # Load the unverified parquet
    logger.info(f"Loading unverified generations from: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    # Determine output path
    if output_parquet is None:
        output_parquet = parquet_file.replace("_NO_VERIFY.parquet", "_VERIFIED.parquet")
    
    logger.info(f"Will save verified results to: {output_parquet}")
    
    # Initialize columns for verification results
    df["success_rate"] = 0.0
    df["generated_solutions_verified"] = None
    
    concurrent_semaphore = threading.Semaphore(max_concurrent_requests)
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying solutions"):
        problem_idx = row["idx"]
        generated_solutions = json.loads(row["generated_solutions"])
        test_cases = json.loads(row["test_cases"])
        
        # Process each rollout for this problem
        verified_solutions = []
        correct_count = 0
        total_count = len(generated_solutions)
        
        for sol in generated_solutions:
            generated_text = sol["text"]
            
            # Extract code if wrapped in markdown
            code_to_test = generated_text
            if "```python" in generated_text:
                code_to_test = generated_text.split("```python")[-1].split("```")[0]
            elif "```" in generated_text:
                parts = generated_text.split("```")
                if len(parts) >= 2:
                    code_to_test = parts[1]
                    if "\n" in code_to_test:
                        first_line, rest = code_to_test.split("\n", 1)
                        if first_line.strip().isalpha():
                            code_to_test = rest
            
            try:
                # Verify using sandbox
                res_list, metadata_list = check_correctness(
                    sandbox_fusion_url=sandbox_fusion_url,
                    in_outs=test_cases,
                    generation=code_to_test,
                    timeout=timeout,
                    memory_limit_mb=memory_limit_mb,
                    language="python",
                    concurrent_semaphore=concurrent_semaphore,
                )
                
                # Score this solution (1 if all tests pass, 0 otherwise)
                score = 1.0 if all(r is True for r in res_list) else 0.0
                correct_count += score
                
                # Record verified solution with score and metadata
                verified_solutions.append({
                    "text": generated_text,
                    "input_tokens": sol.get("input_tokens"),
                    "output_tokens": sol.get("output_tokens"),
                    "input_cost_usd_once": sol.get("input_cost_usd_once"),
                    "output_cost_usd": sol.get("output_cost_usd"),
                    "rollout_cost_usd": sol.get("rollout_cost_usd"),
                    "score": score,
                    "execution_metadata": json.dumps(metadata_list),
                })
            except Exception as e:
                logger.warning(f"Failed to verify solution for problem {problem_idx}: {e}")
                verified_solutions.append({
                    "text": generated_text,
                    "input_tokens": sol.get("input_tokens"),
                    "output_tokens": sol.get("output_tokens"),
                    "input_cost_usd_once": sol.get("input_cost_usd_once"),
                    "output_cost_usd": sol.get("output_cost_usd"),
                    "rollout_cost_usd": sol.get("rollout_cost_usd"),
                    "score": 0.0,
                    "execution_metadata": json.dumps([{"error": str(e)}]),
                })
        
        # Update row with verified results
        df.at[idx, "success_rate"] = correct_count / total_count if total_count > 0 else 0.0
        df.at[idx, "generated_solutions_verified"] = json.dumps(verified_solutions)
    
    # Drop the original unverified column and rename the verified one
    df = df.drop(columns=["generated_solutions"])
    df = df.rename(columns={"generated_solutions_verified": "generated_solutions"})
    
    # Save verified results
    df.to_parquet(output_parquet)
    logger.info(f"✅ Verified results saved to: {output_parquet}")
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total problems verified: {len(df)}")
    logger.info(f"Average success rate: {df['success_rate'].mean():.4f}")
    logger.info(f"Median success rate: {df['success_rate'].median():.4f}")
    logger.info(f"Min success rate: {df['success_rate'].min():.4f}")
    logger.info(f"Max success rate: {df['success_rate'].max():.4f}")
    logger.info("="*60)
    
    return output_parquet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline verification of code generations"
    )
    parser.add_argument(
        "parquet_file",
        type=str,
        help="Path to the NO_VERIFY parquet file from create_code_generations_no_verification.py"
    )
    parser.add_argument(
        "--sandbox-url",
        type=str,
        default=os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080/run_code"),
        help="Sandbox fusion API endpoint (env var: SANDBOX_FUSION_URL)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet file path (default: replace NO_VERIFY with VERIFIED)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Max concurrent sandbox API requests"
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=1024,
        help="Memory limit per execution (MB)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout for each test case (seconds)"
    )
    
    args = parser.parse_args()
    
    verify_offline(
        parquet_file=args.parquet_file,
        sandbox_fusion_url=args.sandbox_url,
        output_parquet=args.output,
        max_concurrent_requests=args.max_concurrent,
        memory_limit_mb=args.memory_limit,
        timeout=args.timeout,
    )
