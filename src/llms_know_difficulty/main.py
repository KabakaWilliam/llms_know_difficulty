import argparse
import torch
from llms_know_difficulty.probe.probe_factory import ProbeFactory
from llms_know_difficulty.utils import (
    create_results_path,
    DataIngestionWorkflow,
    save_probe_predictions
)
from llms_know_difficulty.metrics import compute_metrics
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    
    print("RUNNING WITH CONFIG:")
    print(OmegaConf.to_yaml(cfg))

    print("Loading dataset...")
    # 1. Load the dataset from the config
    # Filter out 'name' field if present (Hydra may add it automatically)
    dataset_cfg = OmegaConf.create({k: v for k, v in cfg.dataset.items() if k != 'name'})
    train_data, val_data, test_data = hydra.utils.instantiate(dataset_cfg, seed=cfg.seed)
    
    print("Creating results directory...")
    # TODO: Now that a different model can be passed to the probe, add this to the results path...
    results_path = create_results_path(root_data_dir=cfg.dataset.root_data_dir,
                                        dataset_name=cfg.dataset.dataset_name,
                                        model_name=cfg.dataset.model_name,
                                        probe_name=cfg.probe.name,
                                        gen_str=None)
    print(f"Creating results directory at {results_path}")

    # 4. Initialize the probe:
    print(f"Initializing probe {cfg.probe.probe_class}\n")
    probe = hydra.utils.instantiate(cfg.probe, device=cfg.device)

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
    results_path = create_results_path(root_data_dir=cfg.dataset.root_data_dir,
                                        dataset_name=cfg.dataset.dataset_name,
                                        model_name=cfg.dataset.model_name,
                                        probe_name=cfg.probe.name,
                                        gen_str=None)
    print(f"Saving probe predictions to {results_path}")
    
    # Already formatted as tensors from predict()
    probe_preds_formatted = [test_indices_tensor, probe_preds_tensor]
    
    save_probe_predictions(probe_preds=probe_preds_formatted,
     probe_metadata=metadata,
     best_probe=probe,
     results_path=results_path)


if __name__ == "__main__":
    main()
