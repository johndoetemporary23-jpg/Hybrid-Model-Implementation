"""
Inference script: Translate single English sentences to Meitei
Usage: python infer.py --text "Hello, how are you?"
"""

import sys
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import HybridTranslationModel

# ================================================================================
# CONFIGURATION
# ================================================================================

CHECKPOINTS_DIR = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================================
# UTILITIES
# ================================================================================

def load_best_model():
    """Load best checkpoint"""
    best_model_path = CHECKPOINTS_DIR / "best_model.pt"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    
    model = HybridTranslationModel()
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    
    return model

def preprocess_text(text, tokenizer, max_length=128):
    """Preprocess English text"""
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return encoding

def ids_to_text(ids, tokenizer):
    """Convert token IDs to text"""
    ids = [id for id in ids if id not in [
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id
    ]]
    return tokenizer.decode(ids, skip_special_tokens=True)

def infer_greedy(model, src_text, beam_size=1):
    """Greedy decoding (faster, single output)"""
    # Preprocess
    encoding = preprocess_text(src_text, model.src_tokenizer)
    src_input_ids = encoding["input_ids"].to(DEVICE)
    src_attention_mask = encoding["attention_mask"].to(DEVICE)
    
    # Generate
    with torch.no_grad():
        pred_ids = model.generate(
            src_input_ids,
            src_attention_mask,
            max_length=128
        )
    
    # Convert to text
    pred_text = ids_to_text(pred_ids[0].cpu().tolist(), model.tgt_tokenizer)
    return pred_text, None

def infer_beam_search(model, src_text, beam_size=5):
    """Beam search decoding (slower, multiple outputs)"""
    # Preprocess
    encoding = preprocess_text(src_text, model.src_tokenizer)
    src_input_ids = encoding["input_ids"].to(DEVICE)
    src_attention_mask = encoding["attention_mask"].to(DEVICE)
    
    # Generate
    with torch.no_grad():
        pred_ids, scores = model.beam_search_generate(
            src_input_ids,
            src_attention_mask,
            beam_size=beam_size,
            max_length=128
        )
    
    # Convert to text
    predictions = []
    for i in range(min(beam_size, pred_ids.size(1))):
        pred_text = ids_to_text(pred_ids[0, i].cpu().tolist(), model.tgt_tokenizer)
        score = scores[0, i].item() if scores is not None else 0.0
        predictions.append((pred_text, score))
    
    return predictions[0][0], predictions

# ================================================================================
# MAIN
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Translate English to Meitei using Hybrid NMT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --text "Hello, how are you?"
  python infer.py --text "What is your name?" --beam_size 5
  python infer.py --text "Good morning" --beam_size 1
        """
    )
    
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="English text to translate"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5, 10],
        help="Beam size for decoding (1=greedy, >1=beam search)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYBRID NMT INFERENCE")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_best_model()
    print(f"✓ Model loaded")
    print()
    
    # Infer
    print(f"Input (English): {args.text}")
    print()
    
    if args.beam_size == 1:
        print("Decoding: Greedy")
        translation, _ = infer_greedy(model, args.text, beam_size=1)
        print(f"Output (Meitei): {translation}")
    else:
        print(f"Decoding: Beam Search (beam_size={args.beam_size})")
        translation, beam_outputs = infer_beam_search(model, args.text, beam_size=args.beam_size)
        
        print(f"\nTop Translation:")
        print(f"  {translation}")
        
        if beam_outputs:
            print(f"\nTop {len(beam_outputs)} Candidates:")
            for idx, (text, score) in enumerate(beam_outputs, 1):
                print(f"  {idx}. (score={score:.4f}) {text}")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
