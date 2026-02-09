"""
Test the PromptingBaseline predictor with MATH dataset.
"""
import sys
sys.path.insert(0, '/VData/linna4335/llms_know_difficult/src')

import pandas as pd
import numpy as np
import re
from pathlib import Path
from pika.probe.prompting_baseline import PromptingBaseline
from pika.config import ROOT_DATA_DIR


def load_math_data():
    """Load MATH dataset with success_rate labels."""
    data_dir = ROOT_DATA_DIR / "SR_DATA" / "DigitalLearningGmbH_MATH-lighteval"
    
    train_file = data_dir / "train-Qwen-Qwen2.5-Math-1.5B-Instruct_maxlen_3000_k_8_temp_0.7.parquet"
    test_file = data_dir / "test-Qwen-Qwen2.5-Math-1.5B-Instruct_maxlen_3000_k_8_temp_0.7.parquet"
    
    print(f"Loading data from {data_dir}")
    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)
    
    # Take subset for quick testing
    train_df = train_df.sample(n=min(100, len(train_df)), random_state=42)
    test_df = test_df.sample(n=min(50, len(test_df)), random_state=42)
    
    print(f"Loaded {len(train_df)} train and {len(test_df)} test samples")
    
    return train_df, test_df, str(train_file)


def extract_gen_params(filename: str) -> dict:
    """Extract generation parameters from filename.
    
    Expected format: ..._{key}_{value}_...
    Example: maxlen_3000_k_8_temp_0.7
    
    Returns dict with 'k', 'maxlen', 'temp', and 'gen_str' (the full suffix)
    """
    # Extract the suffix after the model name
    # Format: ...Instruct_maxlen_3000_k_8_temp_0.7.parquet
    match = re.search(r'(maxlen_\d+_k_\d+_temp_[\d.]+)', filename)
    
    if not match:
        raise ValueError(f"Could not parse generation params from {filename}")
    
    gen_str = match.group(1)
    
    # Parse individual values
    params = {
        'gen_str': gen_str,
    }
    
    # Extract maxlen
    maxlen_match = re.search(r'maxlen_(\d+)', gen_str)
    if maxlen_match:
        params['maxlen'] = int(maxlen_match.group(1))
    
    # Extract k
    k_match = re.search(r'k_(\d+)', gen_str)
    if k_match:
        params['k'] = int(k_match.group(1))
    
    # Extract temp
    temp_match = re.search(r'temp_(\d+\.\d+)', gen_str)
    if temp_match:
        params['temp'] = float(temp_match.group(1))
    
    return params


def main():
    # Configuration
    # MODEL_NAME = "gpt2"
    MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    MODEL_ALIAS = MODEL_NAME.replace("/", "-")
    DATASET_NAME = "DigitalLearningGmbH_MATH-lighteval"
    
    # Generation parameters for VLLM (how to generate the probability answer)
    # Keep deterministic (temp=0.0) so answers are consistent
    GENERATION_TEMPERATURE = 0.0  # Greedy decoding
    GENERATION_MAX_LENGTH = 128
    
    print("="*80)
    print("Testing PromptingBaseline with MATH Dataset")
    print("="*80)
    
    # Load data and extract generation parameters from filename
    train_df, test_df, train_file = load_math_data()
    constraint_params = extract_gen_params(train_file)
    
    print(f"\nExtracted constraint parameters from filename:")
    print(f"  gen_str: {constraint_params['gen_str']}")
    print(f"  k (attempts): {constraint_params['k']}")
    print(f"  maxlen: {constraint_params['maxlen']}")
    print(f"  temp: {constraint_params['temp']}")
    
    # Use extracted parameters for prompt template
    K_VALUE = constraint_params['k']
    MAX_TOKENS = constraint_params['maxlen']
    TEMPERATURE = constraint_params['temp']
    GEN_STR = constraint_params['gen_str']
    
    # Split train into train/val (75/25)
    split_idx = int(0.75 * len(train_df))
    train_data_df = train_df.iloc[:split_idx]
    val_df = train_df.iloc[split_idx:]
    
    train_prompts = train_data_df["formatted_prompt"].tolist()
    train_labels = train_data_df["success_rate"].tolist()
    
    val_prompts = val_df["formatted_prompt"].tolist()
    val_labels = val_df["success_rate"].tolist()
    
    test_prompts = test_df["formatted_prompt"].tolist()
    test_labels = test_df["success_rate"].tolist()
    
    print(f"\nData split:")
    print(f"  Train: {len(train_prompts)} samples")
    print(f"  Val: {len(val_prompts)} samples")
    print(f"  Test: {len(test_prompts)} samples")
    print(f"  Label range: [{min(train_labels + val_labels):.3f}, {max(train_labels + val_labels):.3f}]")
    
    # Initialize and setup probe
    print(f"\n1. Initializing PromptingBaseline...")
    probe = PromptingBaseline(config={})
    
    print(f"2. Setting up VLLM model ({MODEL_NAME})...")
    probe.setup(model_name=MODEL_NAME, device="cuda")
    
    # Train (evaluate) on validation set
    print(f"3. Training on validation set...")
    print(f"   Prompt template will describe solver with: k={K_VALUE}, maxlen={MAX_TOKENS}, temp={TEMPERATURE}")
    print(f"   VLLM will generate answers with: temp={GENERATION_TEMPERATURE}, maxlen={GENERATION_MAX_LENGTH}")
    train_data = (train_prompts, train_labels)
    val_data = (val_prompts, val_labels)
    
    probe.train(
        train_data=train_data,
        val_data=val_data,
        solver_model=MODEL_NAME,
        k=K_VALUE,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    
    # Test on test set
    print(f"\n4. Generating predictions on test set...")
    test_predictions = probe.predict(
        prompts=test_prompts,
        solver_model=MODEL_NAME,
        k=K_VALUE,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    
    # Compute test metric
    test_score, metric_name = probe._compute_metric(
        test_predictions, np.array(test_labels), probe.task_type
    )
    
    print(f"\n{'='*80}")
    print(f"Test {metric_name}: {test_score:.4f}")
    print(f"{'='*80}")
    
    print(f"\nSample predictions vs labels:")
    for i in range(min(5, len(test_prompts))):
        print(f"  Sample {i+1}: Pred={test_predictions[i]:.3f}, Label={test_labels[i]:.3f}")
    
    print(f"\n5. Sample generated text (for debugging):")
    for i in range(min(3, len(probe.val_generations))):
        print(f"\n  --- Generation {i+1} ---")
        gen = probe.val_generations[i]
        # Show first 300 chars or full text if shorter
        print(f"  {gen[:300]}{'...' if len(gen) > 300 else ''}")
    
    # Save results
    print(f"\n6. Saving results...")
    results_path = probe.save_results(
        model_name=MODEL_ALIAS,
        dataset_name=DATASET_NAME,
        gen_str=GEN_STR
    )
    
    # Show what was saved
    print(f"\n7. Results summary:")
    print(f"  Location: {results_path}")
    if (results_path / "val_results.json").exists():
        print(f"  ✓ val_results.json: Validation predictions + {len(probe.val_generations)} generations")
    if (results_path / "test_generations.json").exists():
        print(f"  ✓ test_generations.json: {len(probe.test_generations)} test generations")
    if (results_path / "metadata.json").exists():
        print(f"  ✓ metadata.json: Probe metadata")
    
    print(f"\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()
