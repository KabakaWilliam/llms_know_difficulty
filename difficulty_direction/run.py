import os
import json
import random
import argparse
import logging

import torch
from torchtyping import TensorType
from pathlib import Path

from typing import List, Optional, Union
from .config import Config
from .model_wrapper.list import get_supported_model_class
from .dataset.load_dataset import prepare_all_datasets, load_raw_dataset
from config import DATASET_CONFIGS


from .get_directions_store_preds import generate_directions, select_direction, train_probes
# from .get_directions import generate_directions, select_direction, train_probes
# from .eval import get_refusal_scores, evaluate_jailbreak, evaluate_loss, evaluate_coherence, evaluate_magnitude, compute_magnitude_corr
from .utils import save_to_json_file, load_jsonl, load_parquet

logging.basicConfig(level=logging.INFO)
torch.set_grad_enabled(False);
intervention_labels = ['baseline', 'actadd', 'ablation']



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model')
    parser.add_argument('--config_file', type=str, required=False, default=None, help='Load configuration from file.')
    parser.add_argument('--resume_from_step', type=int, required=False, default=-1, help="Resume from step number")
    parser.add_argument('--run_jailbreak_eval', action="store_true", help="Run jailbreak evaluation")
    parser.add_argument('--run_magnitude_eval', action="store_true", help="Run magnitude evaluation")
    parser.add_argument('--generate_responses', action="store_true", help="Generate responses for datasplits")
    parser.add_argument("--vllm_for_completions", action="store_true", required=False, help="Run E2H-AMC and GSM8K eval with VLLM")
    
    # Cross-validation parameters
    parser.add_argument('--use_k_fold', action="store_true", help="Enable K-Fold cross-validation for probe training")
    parser.add_argument('--n_folds', type=int, default=5, help="Number of folds for K-Fold cross-validation")
    parser.add_argument('--cv_seed', type=int, default=42, help="Random seed for cross-validation")
    
    # Config overrides - these allow modifying config parameters from command line
    parser.add_argument('--n_train', type=int, default=None, help="Number of training samples per dataset")
    parser.add_argument('--n_test', type=int, default=None, help="Number of test samples per dataset")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size for activation extraction")
    parser.add_argument('--generation_batch_size', type=int, default=None, help="Batch size for text generation")
    parser.add_argument('--max_new_tokens', type=int, default=None, help="Maximum new tokens for generation")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save artifacts")
    parser.add_argument('--model_alias', type=str, default=None, help="Alias for the model (used for save path)")
    parser.add_argument('--subset_datasets', type=str, nargs='+', default=None, 
                        help="List of datasets to use for training probes (e.g., --subset_datasets E2H-AMC E2H-GSM8K)")
    parser.add_argument('--evaluation_datasets', type=str, nargs='+', default=None,
                        help="List of datasets to evaluate on (e.g., --evaluation_datasets E2H-AMC E2H-GSM8K)")
    parser.add_argument('--do_sample', action="store_true", help="Use sampling for generation")
    
    return parser.parse_args()


def generate_and_save_candidate_steering_vectors(cfg, DATASET_NAME, model_base, train_set, test_set):
    """Generate and save candidate steering vectors (svs)."""
    os.makedirs(os.path.join(cfg.artifact_path(), f'generate_svs/{DATASET_NAME}'), exist_ok=True)

    train_probes(
        model_base,
        train_data=train_set,
        test_data=test_set,
        artifact_dir=os.path.join(cfg.artifact_path(), f"generate_svs/{DATASET_NAME}"),
        batch_size=cfg.batch_size,
        seed=cfg.cv_seed,
        k_fold=cfg.use_k_fold,
        n_folds=cfg.n_folds)

    # torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))


def generate_and_save_completions_for_dataset(
        cfg:Config,
        model,
        instructions: List[str],
        ratings: Optional[List[str]],
        intervention_label: str = 'baseline',
        coeffs: Union[float, List[float], TensorType[-1]] = 1.0, 
        save_path=None):
    """Generate and save completions for a dataset."""

    completions = model.generate_completions(
        instructions, intervention_method=intervention_label, 
        max_new_tokens=cfg.max_new_tokens, batch_size=cfg.generation_batch_size, coeffs=coeffs
    )
    if ratings is None:
        results = [{"prompt": instructions[i], "response": completions[i]} for i in range(len(instructions))]
    else:
        results = [{"prompt": instructions[i], "response": completions[i], "difficulty": ratings[i]} for i in range(len(instructions))]

    if save_path is None:
        return results
    else:
        save_to_json_file(save_path, results)

# def filter_examples(dataset, scores, threshold):
#     # if harm_type == "harmful":
#     #     comparison = lambda x, y: x > y
#     # else:
#     #     comparison = lambda x, y: x < y
#     return [inst for inst, score in zip(dataset, scores) if comparison(score, threshold)]
    
def load_cached_data(cfg, DATASET_NAME, split, threshold=0):
    try:
        # dataset_path = Path(__file__).parent / "dataset" / "processed" / DATASET_NAME / f"{split}.json"
        # data = json.load(open(dataset_path, "r"))

        data = json.load(open(cfg.artifact_path() / f"datasplits/{DATASET_NAME}_{split}.json", "r"))

        prompts = [x["formatted_prompt"] for x in data]
        scores = [x["difficulty"] for x in data]
        return (prompts, scores)
    except:
        raise FileNotFoundError


def simple_extract_activations(prompts, model, best_pos_idx, best_layer, batch_size):
    print(f"🧠 Extracting activations at layer {best_layer}, position {best_pos_idx}")
    from .get_directions import get_activations
    # Extract activations using the same position as the best probe
    positions = [best_pos_idx]
    activations_list = get_activations(model, prompts, positions=positions, batch_size=batch_size)

    # Concatenate all activations
    all_activations = torch.cat(activations_list, dim=1)  # (n_layer, n_total_prompts, n_pos, hidden_size)
    
    # Get activations for the best layer and position
    target_activations = all_activations[best_layer, :, 0, :]  # (n_prompts, hidden_size)
    
    print(f"✅ Extracted activations: {target_activations.shape}")
    return target_activations



def label_data_with_probe(prompts_list, best_probe, model, best_pos_idx, best_layer, batch_size, already_formatted=False):
    if already_formatted:
        # Prompts are already formatted, use them directly
        prompts = prompts_list
        print(f"📝 Using pre-formatted prompts")
    else:
        # Apply chat template to raw prompts
        prompts = model.apply_chat_template(prompts_list)
        print(f"📝 Applied chat template to prompts")
    
    print(f"📝 Sample prompt: {prompts[0][:200]}...")
    print(f"🧠 Extracting activations at layer {best_layer}, position {best_pos_idx}")
    
    # Get activations for the best layer and position
    target_activations = simple_extract_activations(prompts, model, best_pos_idx, best_layer, batch_size=batch_size)
    
    print(f"✅ Extracted activations: {target_activations.shape}")
    
    # Ensure data types match for matrix multiplication
    target_activations = target_activations.float()
    best_probe = best_probe.float()

    # Calculate difficulty ratings using the best probe direction
    probe_ratings = torch.matmul(target_activations, best_probe)
    print(f"✅ Calculated probe ratings: {probe_ratings.shape}")
    print(f"📊 Rating statistics:")
    print(f"   Mean: {probe_ratings.mean().item():.4f}")
    print(f"   Std: {probe_ratings.std().item():.4f}")
    print(f"   Min: {probe_ratings.min().item():.4f}")
    print(f"   Max: {probe_ratings.max().item():.4f}")

    probe_ratings_sigmoid = torch.sigmoid(probe_ratings)
    print(f"✅ Calculated probe ratings: {probe_ratings.shape}")
    print(f"📊 Rating statistics:")
    print(f"   Mean: {probe_ratings_sigmoid.mean().item():.4f}")
    print(f"   Std: {probe_ratings_sigmoid.std().item():.4f}")
    print(f"   Min: {probe_ratings_sigmoid.min().item():.4f}")
    print(f"   Max: {probe_ratings_sigmoid.max().item():.4f}")

    # Convert tensor to a list of ratings and return it
    return probe_ratings.tolist(), torch.sigmoid(probe_ratings).tolist()
    

def run_pipeline(args):
    """Run the full pipeline."""
    if args.config_file is not None:
        cfg = Config.load(args.config_file)
        # Override config with command line arguments if provided
        if hasattr(args, 'use_k_fold') and args.use_k_fold:
            cfg.use_k_fold = args.use_k_fold
        if hasattr(args, 'n_folds'):
            cfg.n_folds = args.n_folds
        if hasattr(args, 'cv_seed'):
            cfg.cv_seed = args.cv_seed
    else:
        model_alias = os.path.basename(args.model_path)
        
        # Build config with defaults, then apply any command-line overrides
        cfg = Config(
            model_path=args.model_path, 
            model_alias=model_alias,
            use_k_fold=args.use_k_fold if hasattr(args, 'use_k_fold') else False,
            n_folds=args.n_folds if hasattr(args, 'n_folds') else 5,
            cv_seed=args.cv_seed if hasattr(args, 'cv_seed') else 42
        )
        
        # Apply command-line overrides for config parameters
        if args.n_train is not None:
            cfg.n_train = args.n_train
        if args.n_test is not None:
            cfg.n_test = args.n_test
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.generation_batch_size is not None:
            cfg.generation_batch_size = args.generation_batch_size
        if args.max_new_tokens is not None:
            cfg.max_new_tokens = args.max_new_tokens
        if args.save_dir is not None:
            cfg.save_dir = args.save_dir
        if args.subset_datasets is not None:
            cfg.subset_datasets = args.subset_datasets
        if args.evaluation_datasets is not None:
            cfg.evaluation_datasets = args.evaluation_datasets
        if args.do_sample:
            cfg.do_sample = args.do_sample
            
        cfg.save()
    
    # Print configuration summary
    print(f"\n{'='*50}")
    print("Configuration Summary:")
    print(f"{'='*50}")
    print(f"Model: {cfg.model_path}")
    print(f"Model alias: {cfg.model_alias}")
    print(f"Save directory: {cfg.save_dir}")
    print(f"n_train: {cfg.n_train}")
    print(f"n_test: {cfg.n_test}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"generation_batch_size: {cfg.generation_batch_size}")
    print(f"max_new_tokens: {cfg.max_new_tokens}")
    print(f"subset_datasets: {cfg.subset_datasets}")
    print(f"evaluation_datasets: {cfg.evaluation_datasets}")
    print(f"Cross-validation enabled: {cfg.use_k_fold}")
    if cfg.use_k_fold:
        print(f"Number of folds: {cfg.n_folds}")
        print(f"CV seed: {cfg.cv_seed}")
    print(f"{'='*50}\n")
    
    print(f"loading model: {cfg.model_path}")
    model = get_supported_model_class(cfg.model_path)


    datasets = None

    # 0. Load and sample train/valid splits
    print("Loading and validating data splits !")
    if args.resume_from_step <= 0:
        DATASET_NAMES = cfg.subset_datasets
        datasets = prepare_all_datasets(dataset_names=DATASET_NAMES, n_train=cfg.n_train, n_test=cfg.n_test, raw_cfg=cfg)

        save_dir = cfg.artifact_path() / 'datasplits'
        os.makedirs(save_dir, exist_ok=True)

        for DATASET_NAME in DATASET_NAMES:
            for split in ["train", "test"]:
                data = datasets[DATASET_NAME][split]
                INSTRUCTIONS = data["formatted_prompt"].to_list()
                FORMATTED_INSTRUCTIONS = model.apply_chat_template(INSTRUCTIONS)
                RATINGS = data["difficulty"].to_list()

                if args.generate_responses:
                    print(f"generating baseline completions for {DATASET_NAME}: [{split}]")
                    # Generate completions on train/val splits
                    results = generate_and_save_completions_for_dataset(cfg, model, instructions=FORMATTED_INSTRUCTIONS, ratings=RATINGS, intervention_label='baseline')
                else:
                    print(f"preparing datasplit for {DATASET_NAME}: [{split}] (without response generation)")
                    # Create results structure without generating responses
                    results = [{"prompt": FORMATTED_INSTRUCTIONS[i], "difficulty": RATINGS[i]} for i in range(len(FORMATTED_INSTRUCTIONS))]

                # results is a list of dicts. each index is for one response
                # we need to add some metadata
                for idx, r in enumerate(results):
                    r["answer"] = data["answer"].to_list()[idx]
                    r["formatted_prompt"] = FORMATTED_INSTRUCTIONS[idx]
                    r["problem"] = data["problem"].to_list()[idx]
                    r["difficulty"] = data["difficulty"].to_list()[idx]
                    r["uid"] = data["uid"].to_list()[idx]
                
                # if datasets != None :
                #     datasets[DATASET_NAME][split] = results.to_dict()

                save_to_json_file(cfg.artifact_path() / f"datasplits/{DATASET_NAME}_{split}.json", results)
        
        if args.generate_responses:
            print("Generations completed 🎉")
        else:
            print("Datasplits prepared without response generation 🎉")

    # 1. Get activations to train probes
    if args.resume_from_step <= 1:
        logging.info("Extracting activations to train probes")
        DATASET_NAMES = cfg.subset_datasets
        for DATASET_NAME in DATASET_NAMES:
            if datasets is None:
                train_data = load_cached_data(cfg,
                                               DATASET_NAME=DATASET_NAME,
                                               split="train")
                test_data = load_cached_data(cfg,
                                               DATASET_NAME=DATASET_NAME,
                                               split="test")
            else:
                # Convert dataset objects to the expected tuple format
                train_dataset = datasets[DATASET_NAME]["train"]
                test_dataset = datasets[DATASET_NAME]["test"]
                
                # Use the already formatted prompts - no need to apply chat template again
                train_instructions = train_dataset["formatted_prompt"].to_list()
                train_ratings = train_dataset["difficulty"].to_list()
                train_data = (train_instructions, train_ratings)
                
                test_instructions = test_dataset["formatted_prompt"].to_list()
                test_ratings = test_dataset["difficulty"].to_list()
                test_data = (test_instructions, test_ratings)

                print("===================\n")
                print(f"length train set b4 probe: {len(train_ratings)}")
                print(f"length test set b4 probe: {len(test_ratings)}")
                print("===================\n")
            
            generate_and_save_candidate_steering_vectors(cfg, DATASET_NAME, model, train_data, test_data)



    # 2. Get candidate steering vectors
    if args.resume_from_step <= 2:
        logging.info("Evaluating candidate steering vectors")
        DATASET_NAMES = cfg.subset_datasets
        for DATASET_NAME in DATASET_NAMES:
            candidate_svs = torch.load(cfg.artifact_path() / f'generate_svs/{DATASET_NAME}/probe_directions.pt')

            if datasets is None:
                train_data = load_cached_data(cfg,
                                               DATASET_NAME=DATASET_NAME,
                                               split="train")
                test_data = load_cached_data(cfg,
                                               DATASET_NAME=DATASET_NAME,
                                               split="test")
            else:
                # Convert dataset objects to the expected tuple format
                train_dataset = datasets[DATASET_NAME]["train"]
                test_dataset = datasets[DATASET_NAME]["test"]
                
                # Use the already formatted prompts - no need to apply chat template again
                train_instructions = train_dataset["formatted_prompt"].to_list()
                train_ratings = train_dataset["difficulty"].to_list()
                train_data = (train_instructions, train_ratings)

                
                test_instructions = test_dataset["formatted_prompt"].to_list()
                test_ratings = test_dataset["difficulty"].to_list()
                test_data = (test_instructions, test_ratings)
            


            # Load performance metrics to find the best performing probe
            performance_path = cfg.artifact_path() / f'generate_svs/{DATASET_NAME}/probe_performance.json'
            direction_metadata = json.load(open(performance_path, "r"))
            
            # Find the best performing position and layer based on test performance
            best_performance = -float('inf')
            best_pos_idx = 0
            best_layer = 0
            
            layer_performance = direction_metadata["layer_performance"]
            positions = direction_metadata["positions"]
            
            # Search through all positions and layers to find the best test performance
            for pos_idx, pos in enumerate(positions):
                for layer_idx in range(len(layer_performance["test"][pos_idx])):
                    test_perf = layer_performance["test"][pos_idx][layer_idx]
                    if not (test_perf != test_perf):  # Check if not NaN
                        if test_perf > best_performance:
                            best_performance = test_perf
                            best_pos_idx = pos_idx
                            best_layer = layer_idx
            
            print(f"Best probe for {DATASET_NAME}: Position {positions[best_pos_idx]}, Layer {best_layer}, Performance: {best_performance:.4f}")
            
            # Extract the best steering vector
            best_direction = candidate_svs[best_pos_idx, best_layer, :]
            
            # Save the best direction and metadata
            torch.save(best_direction, cfg.artifact_path() / f'generate_svs/{DATASET_NAME}/best_direction.pt')
            
            best_metadata = {
                "dataset": DATASET_NAME,
                "position": positions[best_pos_idx],
                "layer": best_layer,
                "test_performance": best_performance,
                "position_index": best_pos_idx
            }
            
            # Save metadata as a single-item list to match the function signature
            save_to_json_file(cfg.artifact_path() / f'generate_svs/{DATASET_NAME}/best_direction_metadata.json', [best_metadata])
            
            # Set the intervention on the model
            model.set_intervene_direction(best_direction)
            model.set_actAdd_intervene_layer(best_layer)
            
            print(f"Model configured with best steering vector for {DATASET_NAME}")
            print(f"Intervention layer: {best_layer}")
            print(f"Direction shape: {best_direction.shape}")
            print(f"Direction norm: {torch.norm(best_direction).item():.4f}")


            
            print(f"using steering vectors from {DATASET_NAME}")

            # Only run evaluations if responses were generated
            if args.generate_responses:
                EVAL_DATASETS = cfg.evaluation_datasets

                # pr kick my ass if you use aint("eval datasets: ",EVAL_DATASETS)
                # save_dir = cfg.artifact_path() / 'completions'
                # os.makedirs(save_dir, exist_ok=True)

                for eval_dataset_name in EVAL_DATASETS:
                    print("eval datasets: ",EVAL_DATASETS)
                    save_dir = cfg.artifact_path() / f'completions/{eval_dataset_name}'
                    os.makedirs(save_dir, exist_ok=True)

                    print(f"steering generations on {eval_dataset_name} with : {DATASET_NAME} svs")

                    test_data = load_cached_data(
                        cfg,
                        DATASET_NAME=eval_dataset_name,
                        split="test")
                    test_instructions, test_ratings = test_data[0], test_data[1]
                    for intervention in intervention_labels:
                        COEFFS = 0.6
                        # COEFFS_RANGE = [-4, -3, -2, -1, -0.5, 0, +0.5, 1, 2, 3, 4]
                        # COEFFS_RANGE = [6, 7, 8, 9, 10, 11, 12]
                        # COEFFS_RANGE = [-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5]
                        COEFFS_RANGE = [-5, -4, -2, -1, -0.5, 0.5, 1, 4, 5]

                        if intervention == "baseline" or intervention == "ablation":
                            pass
                        else:
                            for COEFF in COEFFS_RANGE:
                                logging.info(f"Evaluating generations {eval_dataset_name} with intervention: {intervention}, coeff: {COEFF}")
                                results = generate_and_save_completions_for_dataset(cfg, model, instructions=test_instructions, ratings=test_ratings, intervention_label=intervention, coeffs=COEFF, save_path=save_dir / f'{DATASET_NAME}_{intervention}_{COEFF}_completions.json')
            else:
                print("Skipping evaluations since --generate_responses was not specified")

    # # #3. Load the best performing probe to add predicted difficulty - each dataset uses its own probe
    # if args.resume_from_step <=3:
    #     # TARGET_DATASETS = ["E2H-AMC", "E2H-GSM8K", "E2H-Codeforces"]  # Datasets to add predicted difficulty to
    #     TARGET_DATASETS = ["E2H-AMC"]  # Datasets to add predicted difficulty to
        
    #     # For each target dataset, use its own probe to make predictions
    #     for target_dataset in TARGET_DATASETS:
    #         print(f"\n🎯 Processing {target_dataset} with its own probe")
            
    #         probe_dir = os.path.join(cfg.artifact_path(), f"generate_svs/{target_dataset}")
    #         directions_file = os.path.join(probe_dir, "probe_directions.pt")
    #         performance_file = os.path.join(probe_dir, "best_direction_metadata.json")
            
    #         # Check if probe exists for this dataset
    #         if not os.path.exists(directions_file) or not os.path.exists(performance_file):
    #             print(f"❌ Probe not found for {target_dataset}, skipping...")
    #             continue
            
    #         try:
    #             # Load performance data
    #             with open(performance_file, 'r') as f:
    #                 performance_data = json.load(f)

    #             # Load probe directions
    #             probe_directions = torch.load(directions_file, map_location='cpu')
    #             print(f"✅ Loaded probe directions: {probe_directions.shape}")

    #             # Find best probe
    #             best_performance = performance_data[0]["test_performance"]
    #             best_pos_idx = performance_data[0]["position"]
    #             best_layer = performance_data[0]["layer"]
                
    #             best_direction = probe_directions[best_pos_idx, best_layer, :]
                
    #             print(f"✅ Best probe for {target_dataset}:")
    #             print(f"   Position: {best_pos_idx} (index {best_pos_idx})")
    #             print(f"   Layer: {best_layer}")
    #             print(f"   Performance: {best_performance:.4f}")
    #             print(f"   Direction shape: {best_direction.shape}")
    #             print(f"   Direction norm: {torch.norm(best_direction).item():.4f}")

    #             target_dataset = "MATH"

    #             # Load the test split data from datasplits directory
    #             test_split_path = cfg.artifact_path() / f"datasplits/{target_dataset}_test.json"
                
    #             if not os.path.exists(test_split_path):
    #                 print(f"❌ Test split file not found: {test_split_path}")
    #                 continue
                
    #             with open(test_split_path, 'r') as f:
    #                 test_data = json.load(f)
                
    #             print(f"✅ Loaded {len(test_data)} items from {test_split_path}")
                
    #             # Extract formatted prompts (already formatted in the saved data)
    #             prompt_list = [item["formatted_prompt"] for item in test_data]
                
    #             # Get predicted difficulty scores using the probe
    #             ratings, ratings_sigmoid = label_data_with_probe(prompt_list, best_direction, model, best_pos_idx, best_layer, batch_size=cfg.batch_size, already_formatted=True)
                
    #             # Add predicted difficulty to each item in the dataset
    #             for i, item in enumerate(test_data):
    #                 # item["predicted_difficulty"] = ratings[i]
    #                 # item["predicted_difficulty_sigmoid"] = ratings_sigmoid[i]
    #                 item[f"predicted_difficulty_{target_dataset}"] = ratings[i]  # Dataset-specific field
    #                 item[f"predicted_difficulty_sigmoid_{target_dataset}"] = ratings_sigmoid[i]
                
    #             # Save the updated data back to the file - using same dataset name for probe since it's its own probe
    #             output_path = cfg.artifact_path() / f"datasplits/{target_dataset}_test_with_predicted_difficulty_{target_dataset}.json"
    #             save_to_json_file(output_path, test_data)
                
    #             print(f"✅ Saved {len(test_data)} items with predicted difficulty to {output_path}")
    #             print(f"📈 Sample predicted difficulties: {ratings[:5]}")
    #             print(f"📈 Sample sigmoid difficulties: {ratings_sigmoid[:5]}")
                
    #         except Exception as e:
    #             print(f"❌ Error processing {target_dataset}: {str(e)}")
    #             continue

    if args.resume_from_step <= 3:
        # unsupervised labelling of other datasets
        PROBING_DATASETS = cfg.subset_datasets
        TARGET_DATASETS = cfg.evaluation_datasets
        for PROBE_NAME in PROBING_DATASETS:
            for target_dataset in TARGET_DATASETS:
                print(f"\n🎯 Labelling {target_dataset} with {PROBE_NAME} probe")
                probe_dir = os.path.join(cfg.artifact_path(), f"generate_svs/{PROBE_NAME}")
                directions_file = os.path.join(probe_dir, "probe_directions.pt")
                performance_file = os.path.join(probe_dir, "best_direction_metadata.json")

                # Check if probe exists for this dataset
                if not os.path.exists(directions_file) or not os.path.exists(performance_file):
                    print(f"❌ Probe not found for {target_dataset}, skipping...")
                    continue
                
                try:
                    # Load performance data
                    with open(performance_file, 'r') as f:
                        performance_data = json.load(f)
                    
                    # Load probe directions
                    probe_directions = torch.load(directions_file, map_location='cpu')
                    print(f"✅ Loaded probe directions: {probe_directions.shape}")

                    # Find best probe
                    best_performance = performance_data[0]["test_performance"]
                    best_pos_idx = performance_data[0]["position"]
                    best_layer = performance_data[0]["layer"]
                    
                    best_direction = probe_directions[best_pos_idx, best_layer, :]
                    
                    print(f"✅ Best probe for {target_dataset}:")
                    print(f"   Position: {best_pos_idx} (index {best_pos_idx})")
                    print(f"   Layer: {best_layer}")
                    print(f"   Performance: {best_performance:.4f}")
                    print(f"   Direction shape: {best_direction.shape}")
                    print(f"   Direction norm: {torch.norm(best_direction).item():.4f}")

                    # Load the test split data from datasplits directory
                    if DATASET_CONFIGS[target_dataset]["dataset_type"] == "local":
                        test_split_path = cfg.artifact_path() / f"datasplits/{target_dataset}_test.json"
                        if not os.path.exists(test_split_path):
                            print(f"❌ Test split file not found: {test_split_path}")
                            continue
                        
                        with open(test_split_path, 'r') as f:
                            test_data = json.load(f)
                        
                        print(f"✅ Loaded {len(test_data)} items from {test_split_path}")
                        
                        # Extract formatted prompts (already formatted in the saved data)
                        prompt_list = [item["formatted_prompt"] for item in test_data]
                    else:
                        test_data = load_raw_dataset(dataset_name=target_dataset, split="test", save_locally=False, raw_cfg=cfg)
                        prompt_col = DATASET_CONFIGS[target_dataset]["prompt_column"]
                        answer_col = DATASET_CONFIGS[target_dataset]["answer_column"]

                        prompt_list = test_data[prompt_col]
                                        # Get predicted difficulty scores using the probe
                        ratings, ratings_sigmoid = label_data_with_probe(prompt_list, best_direction, model, best_pos_idx, best_layer, batch_size=cfg.batch_size, already_formatted=True)
                        
                        # Add predicted difficulty to each item in the dataset
                        for i, item in enumerate(test_data):
                            # item["predicted_difficulty"] = ratings[i]
                            # item["predicted_difficulty_sigmoid"] = ratings_sigmoid[i]
                            item[f"predicted_difficulty"] = ratings[i]  # Dataset-specific field
                            item[f"predicted_difficulty_sigmoid"] = ratings_sigmoid[i]

                        

                        output_path = cfg.artifact_path() / f"datasplits/{target_dataset}_predicted_by_{PROBE_NAME}.json"

                        print(f"new data format\n: {test_data}")

                        if DATASET_CONFIGS[target_dataset]["dataset_type"] == "huggingface":

                            test_data = test_data.add_column("predicted_difficulty", ratings)
                            test_data = test_data.add_column("predicted_difficulty_sigmoid", ratings_sigmoid)

                            final_test_data = [{"question": item["question"], "answer": item["answer"], "predicted_difficulty": item["predicted_difficulty"], "predicted_difficulty_sigmoid": item["predicted_difficulty_sigmoid"]} for item in test_data]


                            with open(output_path, "w") as f:
                                json.dump(final_test_data, f, indent=2)
                        else:
                            save_to_json_file(output_path, test_data)
                        
                        print(f"✅ Saved {len(test_data)} items with predicted difficulty to {output_path}")
                        print(f"📈 Sample predicted difficulties: {ratings[:5]}")
                        print(f"📈 Sample sigmoid difficulties: {ratings_sigmoid[:5]}")


                except Exception as e:
                    print(f"❌ Error processing {target_dataset}: {str(e)}")
                    continue
        return

if __name__ == "__main__":
    args = parse_arguments()

    run_pipeline(args)

