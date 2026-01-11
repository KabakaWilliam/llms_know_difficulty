import argparse
import torch
from probe.probe_factory import ProbeFactory
from utils import (
    create_results_path,
    DataIngestionWorkflow,
    save_probe_predictions
)
from llms_know_difficulty.metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser(description="LLMs-Know-Difficulty command line interface")
    parser.add_argument("--probe", type=str, required=False, help="Name of probe to use")
    parser.add_argument("--dataset", type=str, required=False, help="Path to data file")
    parser.add_argument("--model", type=str, required=False, help="Name of model to use")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to checkpoint file")
    parser.add_argument("--max_len", type=int, required=False, help="Maximum length of the response")
    parser.add_argument("--k", type=int, required=False, help="Number of rollouts per question")
    parser.add_argument("--temperature", type=float, required=False, help="Temperature for the model")
    args = parser.parse_args()

    print("Args:", args)

    # 1. Load the dataset from the config
    train_data, val_data, test_data = DataIngestionWorkflow.load_dataset(
        dataset_name=args.dataset,
        model_name=args.model,
        max_len=args.max_len,
        k=args.k,
        temperature=args.temperature)
    
    # 2. Setup the various metrics we're using -> TODO

    # 3. Setup the results directory for the run 
    results_path = create_results_path(args.dataset, args.model, args.probe)
    print(f"Creating results directory at {results_path}")

    # 4. Initialize the probe:
    print(f"Initializing probe {args.probe}\n")

    probe = ProbeFactory.create_probe(probe_name=args.probe, 
                                        model=args.model,
                                        dataset=args.dataset,
                                        max_len=args.max_len,
                                        k=args.k,
                                        temperature=args.temperature)

    if args.checkpoint_path is not None:
        print(f"Loading probe from checkpoint {args.checkpoint_path}")
        probe.init_model(checkpoint_path=args.checkpoint_path)

    # 6. Run probe training:
    print(f"\n🔥 Training probe on train and val data\n")
    probe = probe.train(train_data=train_data, val_data=val_data)
    
    if probe is None:
        raise RuntimeError("Failed to train probe")

    # 7. Run probe prediction on the test set:
    print(f"Predicting on test data")
    test_indices_tensor, probe_preds_tensor = probe.predict(test_data)

    # 8. Extract labels and compute metrics
    test_labels = list(test_data[2])  # Extract labels from (indices, prompts, targets)
    test_metrics = compute_metrics(test_labels, probe_preds_tensor, task_type=probe.task_type, full_metrics=True)

    # Merge probe metadata with test metrics
    metadata = {
        'best_layer_idx': probe.best_layer_idx,
        'best_pos_idx': probe.best_pos_idx,
        'best_position_value': probe.best_position_value,
        'best_alpha': probe.best_alpha,
        'best_val_score': probe.best_val_score,
        'test_score': probe.test_score,  # Add test score
        'model_name': probe.model_name,
        'd_model': probe.d_model,
        'task_type': probe.task_type,
    }
    # Merge in all test metrics (mse, mae, spearman/auc, acc_all, learnability, precision/recall/f1)
    metadata.update(test_metrics)

    print(f"Test performance: {test_metrics.get('spearman', test_metrics.get('auc', 'N/A'))} 🔥\n")
    # 9. Save the probe predictions to the results directory:
    results_path = create_results_path(args.dataset, args.model, args.probe)
    print(f"Saving probe predictions to {results_path}")
    
    # Already formatted as tensors from predict()
    probe_preds_formatted = [test_indices_tensor, probe_preds_tensor]
    
    save_probe_predictions(probe_preds=probe_preds_formatted,
     probe_metadata=metadata,
     best_probe=probe,
     results_path=results_path)


if __name__ == "__main__":
    main()
