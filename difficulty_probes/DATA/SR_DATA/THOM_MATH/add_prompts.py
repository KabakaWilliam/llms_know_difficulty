from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import os
import glob



# Get all parquet files in current directory
MATH_DATA_DIR = "/VData/linna4335/llms_know_difficult/will_replication/DATA/SR_DATA/THOM_MATH"
ALL_PARQUET_FILES = glob.glob(os.path.join(MATH_DATA_DIR, "MATH_*.parquet"))
print(f"Found {len(ALL_PARQUET_FILES)} parquet files")


PROMPT_SUFFIX = "Let's think step by step and output the final answer within \\boxed{}."

ds = load_dataset("DigitalLearningGmbH/MATH-lighteval")
print(f"Train split: {len(ds['train'])} problems")
print(f"Test split: {len(ds['test'])} problems")

def extract_model_from_filename(filename):
    """Extract the model name from the parquet filename."""
    # Format: MATH_{split}_{samples}-{model}-temperature={temp}.parquet
    basename = os.path.basename(filename)
    parts = basename.split('-')
    
    # Find the part after samples (first part with digits)
    model_parts = []
    found_samples = False
    for part in parts:
        if not found_samples:
            # Check if this part ends with a number (sample size)
            if any(char.isdigit() for char in part):
                # Extract model starting from next part
                found_samples = True
                # But this part might have model start after the number
                after_num = part.split('_')[-1]
                if not after_num[0].isdigit():
                    model_parts.append(after_num)
            continue
        
        # Stop at temperature part
        if 'temperature' in part.lower():
            break
        model_parts.append(part)
    
    model_name = '-'.join(model_parts)
    # Convert back to HF format (e.g., Qwen-Qwen2.5-1.5B-Instruct -> Qwen/Qwen2.5-1.5B-Instruct)
    if model_name.startswith('Qwen-'):
        model_name = 'Qwen/' + model_name[5:]
    
    return model_name


def extract_split_from_filename(filename):
    """Extract the split (train/test) from the parquet filename."""
    basename = os.path.basename(filename)
    if 'train' in basename:
        return 'train'
    elif 'test' in basename:
        return 'test'
    return None


def add_prompts_to_df(df, split, model_name, tokenizer, dataset):
    """Add 'prompt' and 'formatted_prompt' columns to dataframe."""
    n_samples = len(df)
    
    # Get raw problems from MATH dataset
    raw_prompts = dataset[split]["problem"][:n_samples]
    
    # Create suffixed prompts
    suffixed_prompts = [raw_prompt + ' ' + PROMPT_SUFFIX for raw_prompt in raw_prompts]
    
    # Create formatted prompts using chat template
    formatted_prompts = []
    for suffixed_prompt in suffixed_prompts:
        messages = [{"role": "user", "content": suffixed_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)
    
    # Add columns
    df["prompt"] = raw_prompts
    df["formatted_prompt"] = formatted_prompts
    
    return df

# Process all parquet files
tokenizers_cache = {}  # Cache tokenizers to avoid reloading

for i, parquet_file in enumerate(ALL_PARQUET_FILES):
    print(f"\n[{i+1}/{len(ALL_PARQUET_FILES)}] Processing: {os.path.basename(parquet_file)}")
    
    # Extract metadata from filename
    model_name = extract_model_from_filename(parquet_file)
    split = extract_split_from_filename(parquet_file)
    
    if split is None:
        print(f"  ⚠️  Could not determine split, skipping")
        continue
    
    print(f"  Model: {model_name}")
    print(f"  Split: {split}")
    
    # Load dataframe
    df = pd.read_parquet(parquet_file)
    print(f"  Rows: {len(df)}")
    
    # Check if prompts already exist
    if "prompt" in df.columns and "formatted_prompt" in df.columns:
        print(f"  ✓ Prompts already exist, skipping")
        continue
    
    # Load or get cached tokenizer
    if model_name not in tokenizers_cache:
        print(f"  Loading tokenizer for {model_name}...")
        try:
            tokenizers_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"  ❌ Error loading tokenizer: {e}")
            continue
    
    tokenizer = tokenizers_cache[model_name]
    
    # Add prompts
    df = add_prompts_to_df(df, split, model_name, tokenizer, ds)
    
    # Save back to parquet
    print(f"  Saving updated parquet...")
    df.to_parquet(parquet_file, index=False)
    print(f"  ✓ Done")

print(f"\n{'='*60}")
print(f"Finished processing {len(ALL_PARQUET_FILES)} files")
print(f"{'='*60}")