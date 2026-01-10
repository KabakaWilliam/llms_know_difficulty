import argparse
from probe.probe_factory import ProbeFactory
from utils import (
    create_results_path,
    DataIngestionWorkflow,
    save_probe_predictions
    )

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
    print(f"Initializing probe {args.probe}")
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
    print(f"Training probe on train and val data")
    probe = probe.train(train_data=train_data, val_data=val_data)

    # 7. Run probe prediction on the test set:
    print(f"Predicting on test data")
    probe_preds = probe.predict(test_data=test_data)

    # 8. Save the probe predictions to the results directory:
    results_path = create_results_path(args.dataset, args.model, args.probe)
    print(f"Saving probe predictions to {results_path}")
    save_probe_predictions(probe_preds=probe_preds,
     probe_metadata=probe.get_metadata(),
     best_probe=probe,
     results_path=results_path)


if __name__ == "__main__":
    main()
