import argparse
import torch
from llms_know_difficulty.probe.probe_factory import ProbeFactory
from llms_know_difficulty.utils import (
    create_results_path,
    DataIngestionWorkflow,
    save_probe_predictions
)
from llms_know_difficulty.metrics import compute_metrics
from llms_know_difficulty.config import PROMPT_COLUMN_NAME, LABEL_COLUMN_NAME

def main():
    parser = argparse.ArgumentParser(description="LLMs-Know-Difficulty command line interface")
    parser.add_argument("--probe", type=str, required=False, help="Name of probe to use")
    parser.add_argument("--dataset", type=str, required=False, help="Path to data file")
    parser.add_argument("--model", type=str, required=False, help="Name of model to use")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to checkpoint file")
    parser.add_argument("--max_len", type=int, required=False, help="Maximum length of the response")
    parser.add_argument("--k", type=int, required=False, help="Number of rollouts per question")
    parser.add_argument("--temperature", type=float, required=False, help="Temperature for the model")
    parser.add_argument("--prompt_column", type=str, default=PROMPT_COLUMN_NAME, help="Name of prompt column")
    parser.add_argument("--label_column", type=str, default=LABEL_COLUMN_NAME, help="Name of label column")
    args = parser.parse_args()

    print("Args:", args)

    # 1. Load the dataset from the config
    train_data, val_data, test_data = DataIngestionWorkflow.load_dataset(
        dataset_name=args.dataset,
        model_name=args.model,
        max_len=args.max_len,
        k=args.k,
        temperature=args.temperature,
        prompt_column=args.prompt_column,
        label_column=args.label_column)
    
    # 3. Setup the results directory for the run 
    gen_str = f"maxlen_{args.max_len}_k_{args.k}_temp_{args.temperature}"
    results_path = create_results_path(args.dataset, args.model, args.probe, gen_str=gen_str, label_column=args.label_column)
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

    # 8. Extract labels - use only the labels for indices that were actually predicted
    # (in case test_mode limited the predictions)
    all_test_labels = test_data[2]
    # Get labels by position (predictions are returned in order, first N labels match first N predictions)
    test_labels = all_test_labels[:len(test_indices_tensor)]
    test_metrics = compute_metrics(test_labels, probe_preds_tensor, full_metrics=True)
    metadata = probe.get_probe_metadata()  # Get probe-specific metadata
    metadata.update(test_metrics)  # Add test metrics
    
    # Add test_score (main evaluation metric) if available from probe
    if hasattr(probe, 'test_score') and probe.test_score is not None:
        metadata['test_score'] = probe.test_score
    elif 'spearman' in test_metrics:
        metadata['test_score'] = test_metrics['spearman']
    elif 'auc' in test_metrics:
        metadata['test_score'] = test_metrics['auc']

    print(f"Test performance: {test_metrics.get('spearman', test_metrics.get('auc', 'N/A'))} 🔥\n")
    # 9. Save the probe predictions to the results directory:
    # results_path = create_results_path(args.dataset, args.model, args.probe, gen_str=gen_str)
    print(f"Saving probe predictions to {results_path}")
    
    # Already formatted as tensors from predict()
    probe_preds_formatted = [test_indices_tensor, probe_preds_tensor]
    
    save_probe_predictions(probe_preds=probe_preds_formatted,
     probe_metadata=metadata,
     best_probe=probe,
     results_path=results_path)


if __name__ == "__main__":
    main()
