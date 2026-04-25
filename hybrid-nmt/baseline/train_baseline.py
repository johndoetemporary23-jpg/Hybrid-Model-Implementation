"""
Baseline: Fine-tune IndicTrans2 as a complete end-to-end model for comparison
Uses HuggingFace Seq2SeqTrainer for simplicity
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent))

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG_PATH = "config.yaml"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DECODER_MODEL = config["decoder_model"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
SRC_LANG = config["src_lang"]
TGT_LANG = config["tgt_lang"]
MAX_SRC_LEN = config["max_src_len"]
MAX_TGT_LEN = config["max_tgt_len"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Model: {DECODER_MODEL}")
print(f"Source language: {SRC_LANG}")
print(f"Target language: {TGT_LANG}")

# ================================================================================
# UTILITIES
# ================================================================================

def load_pt_data(pt_path):
    """Load .pt file and convert to HuggingFace Dataset format"""
    data = torch.load(pt_path)
    
    # Convert to list of dicts for HuggingFace Dataset
    dataset_dict = {
        "src_input_ids": [],
        "src_attention_mask": [],
        "tgt_input_ids": [],
        "tgt_attention_mask": [],
    }
    
    for sample in data:
        dataset_dict["src_input_ids"].append(sample["src_input_ids"].tolist())
        dataset_dict["src_attention_mask"].append(sample["src_attention_mask"].tolist())
        dataset_dict["tgt_input_ids"].append(sample["tgt_input_ids"].tolist())
        dataset_dict["tgt_attention_mask"].append(sample["tgt_attention_mask"].tolist())
    
    return Dataset.from_dict(dataset_dict)

def preprocess_function(examples, tokenizer=None):
    """Prepare examples for training (for Seq2SeqTrainer compatibility)"""
    # For IndicTrans2, we already have tokenized data
    # Just return as-is but rename for trainer compatibility
    return {
        "input_ids": examples["src_input_ids"],
        "attention_mask": examples["src_attention_mask"],
        "labels": examples["tgt_input_ids"],
    }

# ================================================================================
# TRAINING
# ================================================================================

def train_baseline():
    print("=" * 80)
    print("BASELINE: IndicTrans2 Fine-tuning")
    print("=" * 80)
    print()
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(DECODER_MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL, trust_remote_code=True)
    print(f"✓ Model loaded")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token ID: {tokenizer.pad_token_id}")
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_data = load_pt_data(DATA_DIR / "splits" / "train.pt")
    val_data = load_pt_data(DATA_DIR / "splits" / "val.pt")
    test_data = load_pt_data(DATA_DIR / "splits" / "test.pt")
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print()
    
    # Preprocess
    print("Preprocessing data...")
    train_data = train_data.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=["src_input_ids", "src_attention_mask", "tgt_input_ids", "tgt_attention_mask"]
    )
    val_data = val_data.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=["src_input_ids", "src_attention_mask", "tgt_input_ids", "tgt_attention_mask"]
    )
    test_data = test_data.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=["src_input_ids", "src_attention_mask", "tgt_input_ids", "tgt_attention_mask"]
    )
    print("✓ Data preprocessed")
    print()
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/baseline",
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        fp16=True,
        logging_steps=100,
        logging_dir="./logs/baseline",
        report_to=["tensorboard"],
        generate_summary_metrics=True,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    print("✓ Training complete")
    print()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_data, max_length=MAX_TGT_LEN)
    print(f"Test loss: {test_results['eval_loss']:.4f}")
    
    # Save results
    results = {
        "model": "IndicTrans2 (end-to-end)",
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "test_loss": test_results.get("eval_loss", None),
    }
    
    results_path = RESULTS_DIR / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print()
    
    print("=" * 80)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    train_baseline()
