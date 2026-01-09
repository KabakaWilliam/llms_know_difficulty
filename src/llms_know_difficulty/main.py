import argparse
from config import DATASETS
from utils import create_results_path
from probe.probe_factory import ProbeFactory


def main():
    parser = argparse.ArgumentParser(description="LLMs-Know-Difficulty command line interface")
    parser.add_argument("--probe", type=str, required=False, help="Name of probe to use")
    parser.add_argument("--dataset", type=str, required=False, help="Path to data file")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to checkpoint file")
    args = parser.parse_args()

    print("Args:", args)

    assert args.dataset in DATASETS, f"Dataset {args.dataset} not found in {DATASETS}"

    # 1. Load the dataset from the config
    # TODO: Talk to Thom about what this looks like...


    # 2. Setup the various metrics we're using 
    # TODO: Discuss how we pass this to probes ...


    # 3. Setup the results directory for the run 
    results_path = create_results_path(args.dataset, args.model, args.probe)
    print(f"Creating results directory at {results_path}")

    # 4. Initialize the probe:
    probe, setup_args = ProbeFactory.create_probe(probe_name=args.probe)

    # 5. Run probe setup:
    probe.setup(**setup_args)

    if args.checkpoint_path is not None:
        probe.init_model(checkpoint_path=args.checkpoint_path)

    # 6. Run probe training:
    probe.train(train_data, val_data)

    # 7. Run probe prediction on the test set:
    probe_preds = probe.predict(test_data)

    # 8. Save the probe predictions to the results directory:
    save_probe_predictions(probe_preds, results_path)


if __name__ == "__main__":
    main()
