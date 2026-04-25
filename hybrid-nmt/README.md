# Hybrid Neural Machine Translation: IndicBERT v2 + IndicTrans2

A state-of-the-art hybrid neural machine translation model combining **IndicBERT v2 encoder** + **learned projection bridge** + **IndicTrans2 decoder** for **English → Meitei (Bengali script)** translation.

## 📋 Project Structure

```
hybrid-nmt/
├── data/
│   ├── __init__.py
│   ├── raw/              # Original .txt files (english.txt, meitei.txt)
│   ├── processed/        # Intermediate files
│   ├── splits/           # train.pt, val.pt, test.pt (tokenized)
│   ├── preprocess.py     # Data cleaning, normalization, tokenization
│   └── dataset.py        # PyTorch Dataset class
├── models/
│   ├── __init__.py
│   ├── encoder.py        # IndicBERT v2 wrapper (768-dim)
│   ├── projection.py     # Linear bridge (768→1024)
│   ├── decoder.py        # IndicTrans2 decoder-only (1024-dim)
│   └── hybrid.py         # Full integrated model + set_phase() + generate()
├── baseline/
│   └── train_baseline.py # IndicTrans2 end-to-end baseline
├── results/
│   ├── compare.py        # Side-by-side comparison
│   ├── metrics.json      # Hybrid metrics (BLEU, chrF++, COMET)
│   ├── baseline_results.json
│   ├── predictions.txt   # Generated translations
│   └── comparison.tsv    # Reference vs. Prediction side-by-side
├── checkpoints/
│   ├── best_model.pt     # Best checkpoint (validation loss)
│   ├── latest.pt         # Latest checkpoint (for resuming)
│   └── epoch_*.pt        # Checkpoints at every 5 epochs
├── config.yaml           # Hyperparameters
├── requirements.txt      # Python dependencies
├── setup.sh              # Environment setup script
├── train.py              # Training loop (3 phases, mixed precision, early stopping)
├── evaluate.py           # Test set evaluation + metrics
├── infer.py              # Single-sentence inference CLI
└── .gitignore
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd hybrid-nmt
bash setup.sh
```

This will:
- Install all Python dependencies
- Download IndicBERT v2 (MLM-SamTLM)
- Download IndicTrans2 (en-indic, 200M variant)
- Install IndicTransToolkit

### 2. Preprocess Data

```bash
python data/preprocess.py
```

**Input:** `data/raw/english.txt` and `data/raw/meitei.txt` (300k+ parallel sentences)

**Output:** 
- `data/splits/train.pt` (270k samples)
- `data/splits/val.pt` (15k samples)
- `data/splits/test.pt` (15k samples)

**Processing steps:**
- ✓ Filter by word count (3–100 words)
- ✓ Filter by length ratio (max/min < 3.0)
- ✓ Unicode normalization (IndicNLP)
- ✓ Tokenization (separate tokenizers for src/tgt)
- ✓ Deterministic split (seed=42)

### 3. Train Hybrid Model

```bash
python train.py
```

**3-Phase Training Schedule:**

| Phase | Epochs | Training | LR | Goal |
|-------|--------|----------|-----|------|
| 1 | 1–5 | Bridge only | 5e-4 | Stabilize dimension adaptation |
| 2 | 6–10 | Bridge + Top 4 decoder layers | 1e-4 | Adapt decoder to encoder |
| 3 | 11–20 | Bridge + Top 4 decoder + Top 4 encoder | 5e-5 | Full fine-tuning |

**Features:**
- ✓ Mixed precision training (FP16)
- ✓ Gradient accumulation & clipping
- ✓ Linear warmup + cosine annealing scheduler
- ✓ Early stopping (5-epoch patience)
- ✓ Checkpointing (every 5 epochs + best model)
- ✓ BLEU validation per epoch

### 4. Evaluate on Test Set

```bash
python evaluate.py
```

**Outputs:**
- BLEU score (sacrebleu, flores200 tokenization)
- chrF++ score (character-level F-score)
- COMET score (reference-based metric)
- Top 5 best & worst translations by BLEU

**Files saved:**
- `results/predictions.txt` — all translations
- `results/comparison.tsv` — reference vs. prediction pairs
- `results/metrics.json` — raw scores

### 5. Single-Sentence Inference

```bash
# Greedy decoding (fast)
python infer.py --text "Hello, how are you?" --beam_size 1

# Beam search (slower, higher quality)
python infer.py --text "What is your name?" --beam_size 5
```

### 6. Train Baseline & Compare

```bash
# Fine-tune IndicTrans2 (end-to-end) as baseline
python baseline/train_baseline.py

# Generate comparison table
python results/compare.py
```

## 🏗️ Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│  English Input (e.g., "Hello")          │
│  [42 tokens, max_len=128]               │
└──────────────┬──────────────────────────┘
               │
               ▼
      ┌────────────────────┐
      │  IndicBERT v2      │  ← Pre-trained, frozen (except top 4 in Phase 3)
      │  Encoder           │
      │  Output: 768-dim   │
      └────────┬───────────┘
               │
               ▼
      ┌────────────────────┐
      │  Projection Bridge │  ← Linear(768→1024) + LayerNorm + GELU + Dropout
      │  (TRAINED ALL 3)   │     Handles dimension mismatch
      │  Output: 1024-dim  │
      └────────┬───────────┘
               │
               ▼
      ┌────────────────────────────────────┐
      │  IndicTrans2 Decoder (only)        │  ← Pre-trained, frozen (except top 4 in Phase 2/3)
      │  Cross-attention over encoder      │
      │  Output: logits [vocab_size]       │
      └────────┬─────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Meitei Output (e.g., "ꯑꯗꯣ, ꯑꯩ ꯀꯛꯀ?")       │
│  Greedy or Beam Search decoding             │
└──────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Dimension Bridge (768→1024)**
   - IndicBERT outputs 768-dim embeddings
   - IndicTrans2 decoder expects 1024-dim
   - Learned bridge (`ProjectionBridge`) adapts via `Linear + LayerNorm + GELU + Dropout`

2. **No IndicTrans2 Encoder**
   - We extract ONLY the decoder from IndicTrans2
   - Use IndicBERT for source encoding (likely better for English)
   - Cross-attention in decoder attends over projected encoder output

3. **Phase-Based Training**
   - **Phase 1**: Train bridge from scratch with frozen encoder/decoder
   - **Phase 2**: Unfreeze top 4 decoder transformer layers, keep encoder frozen
   - **Phase 3**: Unfreeze top 4 encoder layers, fine-tune full model

4. **Separate Tokenizers**
   - English (src): IndicBERT tokenizer
   - Meitei (tgt): IndicTrans2 tokenizer
   - Never mix—each model expects its specific vocab

## 📊 Configuration

Edit `config.yaml` to adjust:

```yaml
# Models
encoder_model: "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
decoder_model: "ai4bharat/indictrans2-en-indic-dist-200M"

# Languages
src_lang: "eng_Latn"
tgt_lang: "mni_Beng"  # Meitei in Bengali script

# Dimensions
encoder_dim: 768
decoder_dim: 1024

# Hyperparameters
batch_size: 32
learning_rate: 5e-4        # Phase 1
learning_rate_phase2: 1e-4 # Phase 2
learning_rate_phase3: 5e-5 # Phase 3
warmup_steps: 4000
label_smoothing: 0.1
max_grad_norm: 1.0

# Lengths
max_src_len: 128
max_tgt_len: 128

# Training
num_epochs: 20
phase1_epochs: 5
phase2_epochs: 5
phase3_epochs: 10

# Optimization
fp16: true
early_stopping_patience: 5
```

## 📈 Expected Results

**Rough benchmarks** (on 300k English→Meitei corpus):

| Model | BLEU | chrF++ | COMET |
|-------|------|--------|-------|
| Hybrid (IndicBERT+Bridge+IndicTrans2) | ~25–35 | ~40–50 | ~0.50–0.60 |
| Baseline (IndicTrans2 end-to-end) | ~20–30 | ~35–45 | ~0.45–0.55 |

*Actual numbers depend on data quality, target language complexity, and GPU memory constraints.*

## 🔧 Advanced Usage

### Resume Training from Checkpoint

Training automatically resumes from `checkpoints/latest.pt` if it exists:

```bash
python train.py  # Resumes from latest.pt
```

### Use a Specific Checkpoint

```python
from models import HybridTranslationModel

model = HybridTranslationModel()
checkpoint = torch.load("checkpoints/epoch_10.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```

### Custom Inference with Greedy vs. Beam

```python
from models import HybridTranslationModel

model = HybridTranslationModel()
# ... load checkpoint ...

# Greedy (fast)
pred_ids = model.generate(src_input_ids, src_attention_mask, max_length=128)

# Beam search (slower, higher quality)
pred_ids, scores = model.beam_search_generate(
    src_input_ids, 
    src_attention_mask, 
    beam_size=5, 
    max_length=128
)
```

## 📝 Data Format

**Raw data** (`data/raw/`):
- `english.txt` — one English sentence per line
- `meitei.txt` — corresponding Meitei translation (parallel, line-aligned)

**Preprocessed data** (`data/splits/`):
- PyTorch `.pt` files containing tokenized samples
- Each sample: `{src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask}`

## 🐛 Troubleshooting

### OOM (Out of Memory)
- Reduce `batch_size` in `config.yaml` (e.g., 16 or 8)
- Enable gradient accumulation (`gradient_accumulation_steps: 2`)
- Use a smaller model variant

### Slow Training
- Reduce `num_workers` in DataLoader (default: 4)
- Use smaller `max_src_len` / `max_tgt_len`
- Consider `--fp16` mixed precision (should be enabled by default)

### Poor Metrics
- Increase training epochs (extend Phase 3)
- Lower learning rates (especially Phase 2/3)
- Adjust `warmup_steps` (try 2000–8000)
- Increase `unfreeze_{encoder,decoder}_layers` (e.g., 6 instead of 4)

## 📚 References

- [IndicBERT v2](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-Sam-TLM)
- [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
- [IndicNLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [COMET](https://unbabel.github.io/COMET/)

## 📄 License

This project uses pre-trained models from AI4Bharat, which are available under their respective licenses.

---

**Built with ❤️ for low-resource language translation**
