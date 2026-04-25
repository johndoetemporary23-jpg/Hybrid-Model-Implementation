"""
Evaluation script: Compute metrics (BLEU, chrF++, COMET) on test set
"""

import os
import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models import HybridTranslationModel
from data.dataset import TranslationDataset, collate_fn
from torch.utils.data import DataLoader

try:
    from sacrebleu import BLEU, CHRF
    from comet import download_model, load_model as load_comet_model
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: sacrebleu or comet not installed. Metrics computation will be skipped.")

# ================================================================================
# CONFIGURATION
# ================================================================================

CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ================================================================================
# UTILITIES
# ================================================================================

def load_best_model():
    """Load best checkpoint"""
    best_model_path = CHECKPOINTS_DIR / "best_model.pt"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    
    print(f"Loading model from: {best_model_path}")
    
    model = HybridTranslationModel()
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs, phase {checkpoint['phase']})")
    return model

def tokenize_for_metrics(text, tokenizer):
    """Simple tokenization for metrics"""
    return tokenizer.tokenize(text.lower())

def ids_to_text(ids, tokenizer):
    """Convert token IDs to text"""
    # Remove special tokens
    ids = [id for id in ids if id not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]]
    return tokenizer.decode(ids, skip_special_tokens=True)

def generate_predictions(model, test_loader):
    """Generate predictions on test set"""
    print("\nGenerating predictions...")
    
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            src_input_ids = batch["src_input_ids"].to(DEVICE)
            src_attention_mask = batch["src_attention_mask"].to(DEVICE)
            tgt_input_ids = batch["tgt_input_ids"].to(DEVICE)
            
            # Generate predictions
            pred_ids = model.beam_search_generate(
                src_input_ids,
                src_attention_mask,
                beam_size=5,
                max_length=128
            )[0]  # Get first element (generated_ids)
            
            # Convert to text
            for pred_id_seq, ref_id_seq in zip(pred_ids, tgt_input_ids):
                pred_text = ids_to_text(pred_id_seq.cpu().tolist(), model.tgt_tokenizer)
                ref_text = ids_to_text(ref_id_seq.cpu().tolist(), model.tgt_tokenizer)
                predictions.append(pred_text)
                references.append(ref_text)
    
    return predictions, references

def compute_metrics(predictions, references):
    """Compute BLEU, chrF++, and COMET metrics"""
    print("\nComputing metrics...")
    
    results = {}
    
    # BLEU
    try:
        bleu = BLEU(tokenize='flores200')
        bleu_score = bleu.corpus_score(predictions, [references])
        results["bleu"] = bleu_score.score
        print(f"  BLEU: {bleu_score.score:.2f}")
    except Exception as e:
        print(f"  ⚠️  BLEU computation failed: {e}")
    
    # chrF++
    try:
        chrf = CHRF()
        chrf_score = chrf.corpus_score(predictions, [references])
        results["chrf"] = chrf_score.score
        print(f"  chrF++: {chrf_score.score:.2f}")
    except Exception as e:
        print(f"  ⚠️  chrF++ computation failed: {e}")
    
    # COMET
    try:
        comet_model = load_comet_model("Unbabel/wmt22-comet-da")
        comet_data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(predictions, predictions, references)
        ]
        comet_score = comet_model.predict(comet_data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0)
        results["comet"] = comet_score["system_score"]
        print(f"  COMET: {comet_score['system_score']:.2f}")
    except Exception as e:
        print(f"  ⚠️  COMET computation failed: {e}")
    
    return results

def save_results(predictions, references, metrics, output_dir):
    """Save predictions, comparisons, and metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_path = output_dir / "predictions.txt"
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))
    print(f"\n✓ Predictions saved to: {pred_path}")
    
    # Save comparisons (TSV format)
    comp_path = output_dir / "comparison.tsv"
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write("Reference\tPrediction\n")
        for ref, pred in zip(references, predictions):
            f.write(f"{ref}\t{pred}\n")
    print(f"✓ Comparisons saved to: {comp_path}")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")

def print_best_worst(predictions, references, metrics, num_samples=5):
    """Print best and worst translations"""
    if "bleu" not in metrics:
        print("BLEU metric not available. Skipping best/worst analysis.")
        return
    
    print(f"\n{'='*80}")
    print(f"TOP {num_samples} BEST TRANSLATIONS (by BLEU)")
    print(f"{'='*80}")
    
    # Simple BLEU calculation per sample
    from sacrebleu import BLEU
    bleu = BLEU(tokenize='flores200')
    
    scores = []
    for pred, ref in zip(predictions, references):
        score = bleu.sentence_score(pred, [ref])
        scores.append((score.score, pred, ref))
    
    # Top best
    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
    for i, (score, pred, ref) in enumerate(scores_sorted[:num_samples], 1):
        print(f"\n{i}. BLEU: {score:.2f}")
        print(f"   Reference: {ref[:100]}...")
        print(f"   Prediction: {pred[:100]}...")
    
    # Top worst
    print(f"\n{'='*80}")
    print(f"TOP {num_samples} WORST TRANSLATIONS (by BLEU)")
    print(f"{'='*80}")
    
    for i, (score, pred, ref) in enumerate(scores_sorted[-num_samples:], 1):
        print(f"\n{i}. BLEU: {score:.2f}")
        print(f"   Reference: {ref[:100]}...")
        print(f"   Prediction: {pred[:100]}...")

# ================================================================================
# MAIN
# ================================================================================

def main():
    print("=" * 80)
    print("HYBRID NMT EVALUATION")
    print("=" * 80)
    print()
    
    # Load model
    model = load_best_model()
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    # Load test data
    print("\nLoading test data...")
    test_dataset = TranslationDataset(DATA_DIR / "splits" / "test.pt")
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"✓ Test set: {len(test_dataset)} samples")
    
    # Generate predictions
    predictions, references = generate_predictions(model, test_loader)
    print(f"✓ Generated {len(predictions)} predictions")
    
    # Compute metrics
    metrics = compute_metrics(predictions, references)
    
    # Save results
    save_results(predictions, references, metrics, RESULTS_DIR)
    
    # Print best/worst
    print_best_worst(predictions, references, metrics, num_samples=5)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()
