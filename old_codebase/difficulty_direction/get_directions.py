import os
import json
import numpy as np
import math
import random
from tqdm import tqdm
from sklearn.linear_model import Ridge, RidgeClassifier
from scipy.stats import spearmanr
from typing import List, Optional, Tuple
from torchtyping import TensorType
import torch
from difficulty_direction import ModelBase
from sklearn.model_selection import KFold, train_test_split
# from .eval.refusal import get_refusal_scores, plot_refusal_scores
from .utils import ceildiv, chunks, kl_div_fn

def set_seed(seed: int = 42):
    """Set seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_activations(
    model: ModelBase, instructions: List[str], positions: Optional[List[int]] = [-1], batch_size: Optional[int] = 32
) -> List[TensorType["n_pos", "n_layer", "hidden_size"]]:
    """Compute activations across instructions for each token position and layer"""
    layers = list(range(model.n_layers))
    total = ceildiv(len(instructions), batch_size)
    ACTIVATIONS_LIST  = []
    for instruction_batch in tqdm(chunks(instructions, batch_size), total=total, leave=False):
        formatted_prompts = model.apply_chat_template(instructions=instruction_batch)
        inputs = model.tokenizer(formatted_prompts, padding=True, truncation=False, return_tensors="pt")
        activations = model.get_activations(layers, inputs, positions) # (n_layer, n_prompt, n_pos, hidden_size)

        ACTIVATIONS_LIST.append(activations)
    return ACTIVATIONS_LIST    

def get_mean_activations(
    model: ModelBase, instructions: List[str], positions: Optional[List[int]] = [-1], batch_size: Optional[int] = 32
) -> List[TensorType["n_pos", "n_layer", "hidden_size"]]:
    """Compute mean activations across instructions for each token position and layer"""
    layers = list(range(model.n_layers))
    total = ceildiv(len(instructions), batch_size)
    activation_sums = None
    for instruction_batch in tqdm(chunks(instructions, batch_size), total=total, leave=False):
        formatted_prompts = model.apply_chat_template(instructions=instruction_batch)
        inputs = model.tokenizer(formatted_prompts, padding=True, truncation=False, return_tensors="pt")
        activations = model.get_activations(layers, inputs, positions) # (n_layer, n_prompt, n_pos, hidden_size)
        activations = activations.sum(dim=1) # (n_layer, n_pos, hidden_size)
        if activation_sums is None:
            activation_sums = activations
        else:
            activation_sums += activations

    mean_activations = activation_sums / len(instructions)
    mean_activations = mean_activations.permute(1, 0, 2) # (n_pos, n_layer, hidden_size)
    return mean_activations


def get_mean_diff(
    model: ModelBase, harmful_instructions: List[str], harmless_instructions: List[str], 
    positions: Optional[List[int]] = [-1], batch_size: Optional[int] = 32
) -> List[TensorType["n_pos", "n_layer", "hidden_size"]]:
    """Mean activation difference"""
    harmful_mean_acts = get_mean_activations(model, harmful_instructions, positions=positions, batch_size=batch_size)
    harmless_mean_acts = get_mean_activations(model, harmless_instructions, positions=positions, batch_size=batch_size)
    return harmful_mean_acts - harmless_mean_acts


def generate_directions(
    model: ModelBase, harmful_instructions: List[str], harmless_instructions: List[str], 
    artifact_dir: str, batch_size: Optional[int] = 32
) -> List[TensorType["n_pos", "n_layer", "hidden_size"]]:
    """Generate directions from all layers and post instruction token positions"""
    os.makedirs(artifact_dir, exist_ok=True)

    positions = list(range(-len(model.eoi_toks), 0))
    mean_diffs = get_mean_diff(model, harmful_instructions, harmless_instructions, positions=positions, batch_size=batch_size)

    assert mean_diffs.shape == (len(model.eoi_toks), model.n_layers, model.hidden_size)
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")
    return mean_diffs

def train_probes(
    model: ModelBase, train_data: Tuple[List[str], List[float]], test_data: Tuple[List[str], List[float]], 
    artifact_dir: str, batch_size: Optional[int] = 32, seed: int = 42, k_fold:bool = False, n_folds :int = 5
):
    """Train linear probes from all layers and post instruction token positions"""
    # Set seed for reproducibility
    set_seed(seed)
    
    os.makedirs(artifact_dir, exist_ok=True)
    positions = list(range(-len(model.eoi_toks), 0))

    train_instructions, train_ratings = train_data
    test_instructions, test_ratings = test_data

    # Convert ratings to numpy arrays
    train_ratings = np.array(train_ratings)
    test_ratings = np.array(test_ratings)

    # Get activations for all positions and layers
    print("Getting train activations...")
    train_activations_list = get_activations(model, train_instructions, positions, batch_size)
    print("Getting test activations...")
    test_activations_list = get_activations(model, test_instructions, positions, batch_size)
    
    # Concatenate all batches - convert TensorType to regular tensors first
    train_tensors = [torch.as_tensor(act) for act in train_activations_list]
    test_tensors = [torch.as_tensor(act) for act in test_activations_list]
    
    # Each element in the list has shape (n_layer, n_prompt_batch, n_pos, hidden_size)
    train_activations = torch.cat(train_tensors, dim=1)  # (n_layer, n_prompt_total, n_pos, hidden_size)
    test_activations = torch.cat(test_tensors, dim=1)    # (n_layer, n_prompt_total, n_pos, hidden_size)
    
    # Rearrange to (n_prompt, n_layer, n_pos, hidden_size)
    train_activations = train_activations.permute(1, 0, 2, 3)
    test_activations = test_activations.permute(1, 0, 2, 3)
    
    n_prompts_train, n_layers, n_pos, hidden_size = train_activations.shape
    
    # Initialize storage for probe directions
    probe_directions = torch.zeros((n_pos, n_layers, hidden_size), dtype=torch.float32)
    layer_performance = {"train": [], "test": [], "cv": [] if k_fold else None}
    
    # Train probes for each position and layer
    for pos_idx, pos in enumerate(positions):
        print(f"Training probes for position {pos}")
        pos_train_performance = []
        pos_test_performance = []
        pos_cv_performance = [] if k_fold else None
        
        for layer_idx in tqdm(range(n_layers), desc=f"Layer progress for position {pos}"):
            # Extract activations for this layer and position
            # Shape: (n_prompts, hidden_size)
            x_train = train_activations[:, layer_idx, pos_idx, :].cpu().numpy()
            x_test = test_activations[:, layer_idx, pos_idx, :].cpu().numpy()
            
            if k_fold:
                # K-Fold Cross Validation
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                cv_scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x_train)):
                    x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
                    y_train_fold, y_val_fold = train_ratings[train_idx], train_ratings[val_idx]
                    
                    # Train on fold
                    ridge_model_fold = Ridge(alpha=1.0, fit_intercept=False, random_state=seed)
                    ridge_model_fold.fit(x_train_fold, y_train_fold)
                    
                    # Validate on fold
                    y_pred_val = ridge_model_fold.predict(x_val_fold)
                    val_corr_result = spearmanr(y_val_fold, y_pred_val)
                    val_performance = val_corr_result[0] if isinstance(val_corr_result, tuple) else val_corr_result.correlation
                    cv_scores.append(val_performance)
                
                # Store average CV performance
                avg_cv_performance = np.mean(cv_scores)
                pos_cv_performance.append(avg_cv_performance)
            
            # Train final model on full training set
            ridge_model = Ridge(alpha=1.0, fit_intercept=False, random_state=seed)
            ridge_model.fit(x_train, train_ratings)
            
            # Store the probe direction (coefficients)
            probe_directions[pos_idx, layer_idx, :] = torch.tensor(ridge_model.coef_, dtype=torch.float32)
            
            # Evaluate performance on training set
            y_pred_train = ridge_model.predict(x_train)
            train_corr_result = spearmanr(train_ratings, y_pred_train)
            train_performance = train_corr_result[0] if isinstance(train_corr_result, tuple) else train_corr_result.correlation
            pos_train_performance.append(train_performance)
            
            #store CV scores, and performance on holdout set
            y_pred_test = ridge_model.predict(x_test)
            test_corr_result = spearmanr(test_ratings, y_pred_test)
            test_performance = test_corr_result[0] if isinstance(test_corr_result, tuple) else test_corr_result.correlation
            pos_test_performance.append(test_performance)
        
        layer_performance["train"].append(pos_train_performance)
        layer_performance["test"].append(pos_test_performance) #holdout scotes
        if k_fold:
            layer_performance["cv"].append(pos_cv_performance) #cv scores
    
    # Save probe directions and performance
    torch.save(probe_directions, f"{artifact_dir}/probe_directions.pt")
    
    # Save performance metrics
    performance_data = {
        "layer_performance": layer_performance,
        "positions": positions,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "k_fold": k_fold,
        "n_folds": n_folds if k_fold else None
    }
    
    with open(f"{artifact_dir}/probe_performance.json", 'w') as f:
        json.dump(performance_data, f, indent=4)
    
    print(f"Probe directions saved to {artifact_dir}/probe_directions.pt")
    print(f"Performance metrics saved to {artifact_dir}/probe_performance.json")

    # Print summary of best performance for each position
    for pos_idx, pos in enumerate(positions):
        test_scores = np.array(layer_performance["test"][pos_idx])
        # best by test (holdout)
        best_layer_by_test = int(np.nanargmax(test_scores))
        best_test = float(test_scores[best_layer_by_test])

        if k_fold and layer_performance["cv"] is not None:
            cv_scores = np.array(layer_performance["cv"][pos_idx])
            best_layer_by_cv = int(np.nanargmax(cv_scores))
            best_cv = float(cv_scores[best_layer_by_cv])

            # counterpart metrics at the selected layers
            test_at_cv_layer = float(test_scores[best_layer_by_cv])
            cv_at_test_layer = float(cv_scores[best_layer_by_test]) if cv_scores.size == test_scores.size else float("nan")

            print(f"Position {pos}: Best by CV -> layer {best_layer_by_cv}: CV={best_cv:.4f}; Test@that_layer={test_at_cv_layer:.4f}")
            print(f"Position {pos}: Best by Test -> layer {best_layer_by_test}: Test={best_test:.4f}; CV@that_layer={cv_at_test_layer:.4f}")
        else:
            print(f"Position {pos}: Best by Test -> layer {best_layer_by_test}: Test={best_test:.4f}")
    
    # If k-fold was used, also report the single best probe chosen by CV across all positions/layers,
    # and show how it performed on the holdout test set.
    if k_fold and layer_performance["cv"] is not None:
        best_cv_overall = -np.inf
        best_info = None  # (pos_idx, pos, layer_idx)
        for pos_idx, pos in enumerate(positions):
            cv_scores = np.array(layer_performance["cv"][pos_idx])
            if cv_scores.size == 0:
                continue
            layer_idx = int(np.nanargmax(cv_scores))
            score_cv = float(cv_scores[layer_idx])
            if score_cv > best_cv_overall:
                best_cv_overall = score_cv
                best_info = (pos_idx, pos, layer_idx)

        if best_info is not None:
            best_pos_idx, best_pos, best_layer_idx = best_info
            test_scores_for_pos = np.array(layer_performance["test"][best_pos_idx])
            test_at_best_cv = float(test_scores_for_pos[best_layer_idx]) if test_scores_for_pos.size > best_layer_idx else float("nan")

            print("=== SUMMARY: Best probe by CV across all positions/layers ===")
            print(f"Position {best_pos}, Layer {best_layer_idx}")
            print(f"CV score (selected): {best_cv_overall:.4f}")
            print(f"Test score at that probe: {test_at_best_cv:.4f}")
            print("============================================================")
    
    return probe_directions
    
    # # Print summary of best performance for each position
    # for pos_idx, pos in enumerate(positions):
    #     best_layer_test = np.argmax(layer_performance["test"][pos_idx])
    #     best_performance_test = layer_performance["test"][pos_idx][best_layer_test]
        
    #     if k_fold:
    #         print(f"Position {pos}: Best CV performance {best_performance_test:.4f} at layer {best_layer_test} (using CV as test metric)")
    #         # Also show the CV details
    #         best_layer_cv = np.argmax(layer_performance["cv"][pos_idx])
    #         best_performance_cv = layer_performance["cv"][pos_idx][best_layer_cv]
    #         print(f"Position {pos}: Raw CV performance {best_performance_cv:.4f} at layer {best_layer_cv}")
    #     else:
    #         print(f"Position {pos}: Best test performance {best_performance_test:.4f} at layer {best_layer_test}")
    
    # return probe_directions

def filter_fn(
    refusal_score, steering_score, kl_div_score, layer, n_layer, 
    kl_threshold=None, induce_refusal_threshold=None, prune_layer_percentage: Optional[float] = 0.20
) -> bool:
    """Function for filtering directions. Returns True if the direction should be filtered out
    """
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False


def select_direction(
    model: ModelBase, 
    harmful_instructions: List[str], harmless_instructions: List[str], 
    candidate_directions, artifact_dir: str, 
    kl_threshold: Optional[float] = 0.1, # directions larger KL score are filtered out
    induce_refusal_threshold: Optional[float] = 0.0, # directions with a lower inducing refusal score are filtered out
    prune_layer_percentage: Optional[float] = 0.2, # discard the directions extracted from the last 20% of the model
    batch_size: Optional[int] = 32
) -> Tuple[int, int, TensorType["hidden_size"]]:
    """Select the best vector from a set of candidate directions
    For each vector r, we compute the following:
    1) kl_div_score < 0.1: average KL divergence of harmless_instructions at the last token position (with VS without directional ablation of r)
    2) bypass_score: average refusal score across harmful_instructions (with directional ablation of r).
    3) induce_score > 0: average refusal scores across harmless_instructions (with activation addition of r).
    Addionally, the layer of r < (1 - prune_layer_percentage) * n_layer. This ensures r is not too close to the unembedding directions.
    """
    os.makedirs(artifact_dir, exist_ok=True)

    n_pos, n_layer, hidden_size = candidate_directions.shape
    refusal_toks = model.refusal_toks

    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)

    baseline_refusal_scores_harmful = get_refusal_scores(model.get_last_position_logits(harmful_instructions, batch_size=batch_size), refusal_toks=refusal_toks)
    baseline_harmless_logits = model.get_last_position_logits(harmless_instructions, batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(baseline_harmless_logits, refusal_toks=refusal_toks)

    for source_pos in tqdm(range(-n_pos, 0), desc="Token position", leave=False):
        for source_layer in (pbar := tqdm(range(n_layer), leave=False)):
            pbar.set_description(f"Evaluating model layer {source_layer}")
            direction = candidate_directions[source_pos, source_layer]
            
            # Compute kl_score
            intervention_logits = model.get_last_position_logits(harmless_instructions, direction=direction, intervention_method="ablation", batch_size=batch_size)
            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

            # Compute bypass_score
            intervention_logits = model.get_last_position_logits(harmful_instructions, direction=direction, intervention_method="ablation", batch_size=batch_size)
            refusal_scores = get_refusal_scores(intervention_logits, refusal_toks=refusal_toks)
            ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

            # Compute induce_score
            intervention_logits = model.get_last_position_logits(
                harmless_instructions, direction=direction, intervention_method="actadd", 
                steering_layer=source_layer, coeffs=1.0, batch_size=batch_size
            )
            refusal_scores = get_refusal_scores(intervention_logits, refusal_toks=refusal_toks)
            steering_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()


    token_labels = model.tokenizer.convert_ids_to_tokens(model.eoi_toks)
    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=token_labels,
        title=f'Ablating direction on harmful instructions [{model.model_name}]',
        artifact_dir=artifact_dir,
        artifact_name='bypass_scores_ablation'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=token_labels,
        title=f'Adding direction on harmless instructions [{model.model_name}]',
        artifact_dir=artifact_dir,
        artifact_name='induce_scores_actadd'
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=token_labels,
        title=f'KL Divergence when ablating direction on harmless instructions [{model.model_name}]',
        artifact_dir=artifact_dir,
        artifact_name='kl_div_scores_ablation'
    )

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'bypass_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'induce_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

            refusal_score = ablation_refusal_scores[source_pos, source_layer].item()
            steering_score = steering_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            # we sort the directions in descending order (from highest to lowest score)
            # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
            sorting_score = -refusal_score

            # we filter out directions if the KL threshold 
            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'bypass_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'induce_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })   

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['bypass_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Refusal score: {ablation_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"Steering score: {steering_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")
    
    return pos, layer, candidate_directions[pos, layer]
