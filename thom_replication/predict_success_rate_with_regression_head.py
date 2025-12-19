#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# 🔹 import YOUR code
from utils.text_number_dataset import TextNumberDataset
from utils.metrics import compute_metrics
import numpy as np

def hf_compute_metrics(eval_pred):
    preds, labels = eval_pred  # EvalPrediction is tuple-like
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # For regression heads, preds is often shape (N, 1)
    if preds.ndim > 1:
        preds = preds.squeeze(-1)

    ys_t = torch.tensor(labels, dtype=torch.float32)
    preds_t = torch.tensor(preds, dtype=torch.float32)

    return compute_metrics(ys_t, preds_t)


# ----------------------------
# Collation
# ----------------------------
@dataclass
class CollateConfig:
    max_length: int

def make_collate_fn(tokenizer, cfg: CollateConfig):
    def collate(batch: List[Tuple[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        texts, ys = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.stack(list(ys))
        return enc
    return collate


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser("Train HF LLM with regression head")

    # Data
    parser.add_argument("--hf_dataset", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--train_scores_path", required=True)
    parser.add_argument("--val_scores_path", required=True)

    # Model
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=512)

    # Training
    parser.add_argument("--output_dir", default="./regression-llm")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--fp16", action="store_true")

    # python predict_success_rate_with_regression_head.py \
    # --model Qwen/Qwen2-1.5B-Instruct \
    # --hf_dataset DigitalLearningGmbH/MATH-lighteval \
    # --train_scores_path /home/thomfoster/llms_know_difficulty/thom_replication/data/MATH_train-Qwen-Qwen2-1.5B-Instruct.parquet \
    # --val_scores_path /home/thomfoster/llms_know_difficulty/thom_replication/data/MATH_test-Qwen-Qwen2-1.5B-Instruct.parquet \
    # --max_length 512 \
    # --lr 1e-4 \
    # --batch_size 128 \


    args = parser.parse_args()
    set_seed(args.seed)

    # ----------------------------
    # Tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collate_fn = make_collate_fn(
        tokenizer, CollateConfig(max_length=args.max_length)
    )

    # ----------------------------
    # Datasets
    # ----------------------------
    train_ds = TextNumberDataset(
        hf_dataset=args.hf_dataset,
        hf_dataset_split=args.train_split,
        scores_path=args.train_scores_path,
    )
    eval_ds = TextNumberDataset(
        hf_dataset=args.hf_dataset,
        hf_dataset_split=args.eval_split,
        scores_path=args.val_scores_path,
    )
    eval_ds.items = eval_ds.items[:1000]  # limit eval set size for speed

    # ----------------------------
    # Model (regression head)
    # ----------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        problem_type="regression",
    )
    model.resize_token_embeddings(len(tokenizer))

    # ----------------------------
    # Trainer
    # ----------------------------

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="steps",        # ← renamed
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        logging_steps=50,
        report_to="wandb",
    )

    from transformers import TrainerCallback
    class StripEvalPrefixCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for k in list(logs.keys()):
                    if k.startswith("eval/") or k.startswith("eval_"):
                        logs[k[5:]] = logs.pop(k)
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=hf_compute_metrics,
        callbacks=[StripEvalPrefixCallback()],
    )

    trainer.train()
    trainer.evaluate()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
