"""
Data preprocessing pipeline for Hybrid NMT
Steps: load → filter → normalize → tokenize → split → save
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import indic-nlp-library for normalization
try:
    from indicnlp.normalize import indic_normalize
    from indicnlp.transform import indic_translit
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False
    print("Warning: indic-nlp-library not found. Unicode normalization will be limited.")

from transformers import AutoTokenizer

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG_PATH = "config.yaml"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

ENCODER_MODEL = config["encoder_model"]
DECODER_MODEL = config["decoder_model"]
SRC_LANG = config["src_lang"]
TGT_LANG = config["tgt_lang"]
MAX_SRC_LEN = config["max_src_len"]
MAX_TGT_LEN = config["max_tgt_len"]
TRAIN_SIZE = config["train_split"]
VAL_SIZE = config["val_split"]
TEST_SIZE = config["test_split"]

RANDOM_SEED = 42

# ================================================================================
# STEP 1: LOAD AND BASIC CLEANING
# ================================================================================

def load_and_clean(english_path, meitei_path):
    """Load parallel files and perform basic cleaning"""
    print("\n[STEP 1] Loading and basic cleaning...")
    
    with open(english_path, "r", encoding="utf-8") as f:
        eng_lines = [line.strip() for line in f.readlines()]
    
    with open(meitei_path, "r", encoding="utf-8") as f:
        mei_lines = [line.strip() for line in f.readlines()]
    
    print(f"  Loaded: {len(eng_lines)} English, {len(mei_lines)} Meitei lines")
    
    # Remove empty lines and create pairs
    pairs = []
    removal_stats = defaultdict(int)
    
    for eng, mei in zip(eng_lines, mei_lines):
        # Remove empty pairs
        if not eng.strip() or not mei.strip():
            removal_stats["empty_pair"] += 1
            continue
        
        eng_words = eng.split()
        mei_words = mei.split()
        
        # Filter by word count (3 to 100 words)
        if len(eng_words) < 3 or len(eng_words) > 100:
            removal_stats["eng_word_count_violation"] += 1
            continue
        if len(mei_words) < 3 or len(mei_words) > 100:
            removal_stats["mei_word_count_violation"] += 1
            continue
        
        # Filter by length ratio (longer/shorter must be <= 3.0)
        eng_char_len = len(eng)
        mei_char_len = len(mei)
        max_len = max(eng_char_len, mei_char_len)
        min_len = min(eng_char_len, mei_char_len)
        
        if max_len > 0 and max_len / min_len > 3.0:
            removal_stats["length_ratio_violation"] += 1
            continue
        
        pairs.append((eng, mei))
    
    print(f"  After filtering: {len(pairs)} pairs")
    print("  Removal stats:")
    for reason, count in removal_stats.items():
        print(f"    - {reason}: {count}")
    
    return pairs, removal_stats

# ================================================================================
# STEP 2: NORMALIZE WITH INDIC-NLP
# ================================================================================

def normalize_text(text, lang="eng"):
    """Normalize text using indic-nlp-library if available"""
    if not INDIC_NLP_AVAILABLE:
        return text
    
    if lang == "mni":
        try:
            # Normalize Meitei text (mni_Beng = Meitei in Bengali script)
            normalizer = indic_normalize.IndicNormalize(lang="mni")
            text = normalizer.normalize(text)
        except Exception as e:
            # Fallback: just return original
            pass
    
    return text

def normalize_pairs(pairs):
    """Normalize all pairs"""
    print("\n[STEP 2] Normalizing text...")
    
    normalized = []
    for eng, mei in tqdm(pairs, desc="Normalizing"):
        eng = normalize_text(eng, lang="eng")
        mei = normalize_text(mei, lang="mni")
        normalized.append((eng, mei))
    
    return normalized

# ================================================================================
# STEP 3: TOKENIZE
# ================================================================================

def tokenize_pairs(pairs):
    """Tokenize using IndicBERT (src) and IndicTrans2 (tgt) tokenizers"""
    print("\n[STEP 3] Loading tokenizers...")
    
    src_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL, trust_remote_code=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL, trust_remote_code=True)
    
    print(f"  Source tokenizer (IndicBERT): vocab_size={len(src_tokenizer)}")
    print(f"  Target tokenizer (IndicTrans2): vocab_size={len(tgt_tokenizer)}")
    
    # Get language token IDs
    # For IndicBERT, we may need to add language tokens
    # For IndicTrans2, language tokens are typically in the tokenizer
    
    print("\n  Tokenizing pairs...")
    tokenized_pairs = []
    
    for eng, mei in tqdm(pairs, desc="Tokenizing"):
        # Tokenize source (English)
        src_encoding = src_tokenizer(
            eng,
            max_length=MAX_SRC_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize target (Meitei)
        # For IndicTrans2, prepend language token
        tgt_text = f"{TGT_LANG} {mei}"  # Language token prefix
        tgt_encoding = tgt_tokenizer(
            tgt_text,
            max_length=MAX_TGT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized_pairs.append({
            "src_input_ids": src_encoding["input_ids"].squeeze(0),
            "src_attention_mask": src_encoding["attention_mask"].squeeze(0),
            "tgt_input_ids": tgt_encoding["input_ids"].squeeze(0),
            "tgt_attention_mask": tgt_encoding["attention_mask"].squeeze(0),
        })
    
    print(f"  Tokenized {len(tokenized_pairs)} pairs")
    return tokenized_pairs, src_tokenizer, tgt_tokenizer

# ================================================================================
# STEP 4: SPLIT DETERMINISTICALLY
# ================================================================================

def split_data(tokenized_pairs, train_size, val_size, test_size, seed=42):
    """Split data deterministically into train/val/test"""
    print(f"\n[STEP 4] Splitting data (seed={seed})...")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    indices = np.arange(len(tokenized_pairs))
    np.random.shuffle(indices)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:train_size + val_size + test_size]
    
    train_data = [tokenized_pairs[i] for i in train_idx]
    val_data = [tokenized_pairs[i] for i in val_idx]
    test_data = [tokenized_pairs[i] for i in test_idx]
    
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    return train_data, val_data, test_data

# ================================================================================
# STEP 5: SAVE AND COMPUTE STATISTICS
# ================================================================================

def save_splits(train_data, val_data, test_data, output_dir):
    """Save splits as .pt files and compute statistics"""
    print(f"\n[STEP 5] Saving splits to {output_dir}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    torch.save(train_data, output_dir / "train.pt")
    torch.save(val_data, output_dir / "val.pt")
    torch.save(test_data, output_dir / "test.pt")
    
    print(f"  ✓ Saved train.pt, val.pt, test.pt")
    
    # Compute statistics
    def compute_stats(data, split_name):
        src_lengths = []
        tgt_lengths = []
        
        for sample in data:
            # Count non-padding tokens
            src_len = (sample["src_attention_mask"] == 1).sum().item()
            tgt_len = (sample["tgt_attention_mask"] == 1).sum().item()
            src_lengths.append(src_len)
            tgt_lengths.append(tgt_len)
        
        print(f"\n  {split_name}:")
        print(f"    Avg src length: {np.mean(src_lengths):.2f} tokens")
        print(f"    Avg tgt length: {np.mean(tgt_lengths):.2f} tokens")
        print(f"    Max src length: {np.max(src_lengths)} tokens")
        print(f"    Max tgt length: {np.max(tgt_lengths)} tokens")
    
    compute_stats(train_data, "Train")
    compute_stats(val_data, "Val")
    compute_stats(test_data, "Test")

# ================================================================================
# MAIN
# ================================================================================

def main():
    print("=" * 80)
    print("HYBRID NMT DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Source language: {SRC_LANG}")
    print(f"  Target language: {TGT_LANG}")
    print(f"  Max src length: {MAX_SRC_LEN} tokens")
    print(f"  Max tgt length: {MAX_TGT_LEN} tokens")
    print(f"  Encoder model: {ENCODER_MODEL}")
    print(f"  Decoder model: {DECODER_MODEL}")
    
    # Step 1: Load and clean
    eng_file = RAW_DIR / "english.txt"
    mei_file = RAW_DIR / "meitei.txt"
    
    if not eng_file.exists() or not mei_file.exists():
        print(f"\n❌ Error: Data files not found!")
        print(f"  Expected: {eng_file}, {mei_file}")
        sys.exit(1)
    
    pairs, removal_stats = load_and_clean(eng_file, mei_file)
    
    # Step 2: Normalize
    pairs = normalize_pairs(pairs)
    
    # Step 3: Tokenize
    tokenized_pairs, src_tokenizer, tgt_tokenizer = tokenize_pairs(pairs)
    
    # Step 4: Split
    train_data, val_data, test_data = split_data(
        tokenized_pairs,
        TRAIN_SIZE,
        VAL_SIZE,
        TEST_SIZE,
        seed=RANDOM_SEED
    )
    
    # Step 5: Save
    save_splits(train_data, val_data, test_data, SPLITS_DIR)
    
    # Summary
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total pairs before filtering: {len(pairs) + sum(removal_stats.values())}")
    print(f"  Total pairs after filtering: {len(pairs)}")
    print(f"  Pairs in train/val/test: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    print(f"\n  Splits saved to: {SPLITS_DIR}/")
    print(f"    - train.pt ({len(train_data)} samples)")
    print(f"    - val.pt ({len(val_data)} samples)")
    print(f"    - test.pt ({len(test_data)} samples)")
    print()

if __name__ == "__main__":
    main()
