================================================================================
PROMPT 1 — PROJECT STRUCTURE & ENVIRONMENT
================================================================================
 
You are helping me build a hybrid Neural Machine Translation model.
Project: IndicBERT v2 encoder + IndicTrans2 decoder for English to Meitei (mni_Mtei) translation.
Hardware: Lightning AI Studio (GPU). Framework: PyTorch + HuggingFace Transformers.
Corpus: 300,000 parallel sentence pairs in two .txt files (one English, one Meitei).
 
Task: Create the complete project scaffold.
 
1. Create this exact folder structure:
   hybrid-nmt/
   ├── data/
   │   ├── raw/          # original .txt files go here
   │   ├── processed/    # after tokenization
   │   └── splits/       # train / val / test
   ├── models/
   │   ├── encoder.py    # IndicBERT v2 wrapper
   │   ├── decoder.py    # IndicTrans2 decoder wrapper
   │   ├── projection.py # Linear(768->1024) + LayerNorm bridge
   │   └── hybrid.py     # full model combining all three
   ├── train.py
   ├── evaluate.py
   ├── infer.py
   └── config.yaml
 
2. Create a requirements.txt with these exact packages:
   torch, transformers, datasets, sentencepiece, sacrebleu,
   accelerate, indic-nlp-library, unbabel-comet,
   pyyaml, tqdm, numpy
 
3. Create a setup.sh script that:
   - Installs all requirements with pip
   - Downloads ai4bharat/IndicBERTv2-MLM-Sam-TLM using huggingface-cli
   - Downloads ai4bharat/indictrans2-en-indic-dist-200M using huggingface-cli
   - Installs IndicTransToolkit from git+https://github.com/VarunGumma/IndicTransToolkit.git
 
4. Create config.yaml with all hyperparameters:
   encoder_model: ai4bharat/IndicBERTv2-MLM-Sam-TLM
   decoder_model: ai4bharat/indictrans2-en-indic-dist-200M
   src_lang: eng_Latn
   tgt_lang: mni_Mtei
   encoder_dim: 768
   decoder_dim: 1024
   batch_size: 32
   learning_rate: 5e-4
   warmup_steps: 4000
   label_smoothing: 0.1
   max_src_len: 128
   max_tgt_len: 128
   train_split: 270000
   val_split: 15000
   test_split: 15000
   fp16: true
 
================================================================================
PROMPT 2 — DATA PREPROCESSING PIPELINE
================================================================================
 
I have two raw .txt files in data/raw/: english.txt and meitei.txt
Each line is one sentence. They are parallel (line N in english = line N in meitei).
Total: ~300,000 sentence pairs.
 
Task: Write a complete data/preprocess.py script that does the following steps in order:
 
Step 1 - Load and basic clean:
  - Read both files line by line
  - Strip whitespace, remove empty lines
  - Remove pairs where either sentence is empty
  - Remove pairs where length ratio (longer/shorter word count) > 3.0
  - Remove pairs where either sentence has fewer than 3 words or more than 100 words
 
Step 2 - Normalize using IndicNLP:
  - Use indic_nlp_library to normalize the Meitei text
  - Normalize Unicode for both English and Meitei
 
Step 3 - Tokenize:
  - Use IndicBERT tokenizer (ai4bharat/IndicBERTv2-MLM-Sam-TLM) for English source
  - Use IndicTrans2 tokenizer (ai4bharat/indictrans2-en-indic-dist-200M) for Meitei target
  - Add language tokens: src_lang=eng_Latn, tgt_lang=mni_Mtei
  - Max length: 128 tokens for both sides
  - Save tokenized data as .pt (PyTorch tensors)
 
Step 4 - Split deterministically (set random seed=42):
  - Train: 270,000 pairs
  - Validation: 15,000 pairs
  - Test: 15,000 pairs
  - Save to data/splits/train.pt, val.pt, test.pt
 
Step 5 - Print a summary:
  - Total pairs before and after filtering
  - How many were removed and why
  - Average sentence length (tokens) for src and tgt in each split
 
Also write a data/dataset.py with a PyTorch Dataset class called TranslationDataset
that loads the .pt files and returns (src_input_ids, src_attention_mask,
tgt_input_ids, tgt_attention_mask) as tensors.
 
================================================================================
PROMPT 3 — HYBRID MODEL ARCHITECTURE
================================================================================
 
Build the hybrid translation model with these exact 4 files:
 
FILE 1: models/encoder.py
  - Class: IndicBERTEncoder
  - Loads ai4bharat/IndicBERTv2-MLM-Sam-TLM using AutoModel
  - forward() takes (input_ids, attention_mask) and returns
    last_hidden_state of shape [batch, seq_len, 768]
  - Add a freeze() method that freezes all parameters
  - Add an unfreeze_top_layers(n=4) method that unfreezes the top N
    transformer layers only
 
FILE 2: models/projection.py
  - Class: ProjectionBridge
  - Input dim: 768 (IndicBERT output)
  - Output dim: 1024 (IndicTrans2 decoder input)
  - Architecture: Linear(768, 1024) -> LayerNorm(1024) -> GELU -> Dropout(0.1)
  - This is the ONLY component trained from scratch in Phase 1
  - Initialize weights with xavier_uniform_
 
FILE 3: models/decoder.py
  - Class: IndicTrans2Decoder
  - Loads ai4bharat/indictrans2-en-indic-dist-200M using AutoModelForSeq2SeqLM
  - Extract ONLY model.model.decoder from the full model
  - Extract model.lm_head for final token projection
  - forward() takes (tgt_input_ids, encoder_hidden_states, encoder_attention_mask)
    and returns logits of shape [batch, tgt_seq_len, vocab_size]
  - Add freeze() and unfreeze_top_layers(n=4) methods like the encoder
  - Force BOS token id to mni_Mtei language token
 
FILE 4: models/hybrid.py
  - Class: HybridTranslationModel(nn.Module)
  - Combines IndicBERTEncoder + ProjectionBridge + IndicTrans2Decoder
  - forward() takes (src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask)
    and returns logits
  - Add a set_phase(phase: int) method:
      phase=1: freeze encoder and decoder, train only projection
      phase=2: freeze encoder, unfreeze top 4 decoder layers + projection
      phase=3: unfreeze top 4 encoder + top 4 decoder layers + projection
  - Add a generate() method using greedy decoding for inference
  - Add a beam_search_generate() method with beam_size=5
 
IMPORTANT NOTES:
  - IndicBERT output dim is 768, IndicTrans2 decoder expects 1024
  - The projection bridge MUST handle the dimension mismatch
  - Cross-attention in the decoder attends over the projected encoder output
  - Do NOT use IndicTrans2's own encoder at all
 
================================================================================
PROMPT 4 — TRAINING SCRIPT
================================================================================
 
Write a complete train.py script for the HybridTranslationModel.
 
Requirements:
 
1. Load config from config.yaml using PyYAML
 
2. Training phases (implement all 3 automatically in sequence):
   Phase 1 (epochs 1-5):   train only ProjectionBridge, LR=5e-4
   Phase 2 (epochs 6-10):  train ProjectionBridge + top 4 decoder layers, LR=1e-4
   Phase 3 (epochs 11-20): train ProjectionBridge + top 4 decoder + top 4 encoder, LR=5e-5
   Call model.set_phase(n) at the start of each phase
 
3. Loss: CrossEntropyLoss with label_smoothing=0.1
   Ignore padding token index in loss calculation
 
4. Optimizer: AdamW with weight_decay=0.01
   Use separate param groups for each phase with correct LRs
 
5. Scheduler: Linear warmup for 4000 steps then cosine annealing
 
6. Mixed precision: use torch.cuda.amp.GradScaler and autocast
 
7. Gradient clipping: max_norm=1.0
 
8. DataLoader: batch_size=32, num_workers=4, pin_memory=True
 
9. Checkpointing:
   - Save best model based on validation loss to checkpoints/best_model.pt
   - Save checkpoint every 5 epochs to checkpoints/epoch_{n}.pt
   - Resume from checkpoint if checkpoints/latest.pt exists
 
10. Logging:
    - Print train loss, val loss, val BLEU every epoch
    - Log which phase is active and which layers are frozen/unfrozen
    - Use tqdm for progress bars
 
11. Early stopping: stop if val loss does not improve for 5 epochs
 
12. At the end of training print a summary table:
    best epoch, best val loss, best val BLEU for each phase
 
================================================================================
PROMPT 5 — EVALUATION & INFERENCE
================================================================================
 
Write two scripts:
 
SCRIPT 1: evaluate.py
  - Load the best checkpoint from checkpoints/best_model.pt
  - Run inference on data/splits/test.pt
  - Compute and print these 3 metrics:
      BLEU score using sacrebleu (tokenize='flores200')
      chrF++ score using sacrebleu
      COMET score using unbabel-comet (model: Unbabel/wmt22-comet-da)
  - Save all predictions to results/predictions.txt
  - Save a side-by-side comparison (source | reference | prediction)
    to results/comparison.tsv
  - Print top 5 best and worst translations by BLEU score
 
SCRIPT 2: infer.py
  - Load best checkpoint
  - Accept a plain English string from command line argument
  - Preprocess it the same way as training data
  - Run beam search with beam_size=5
  - Print the Meitei translation
  - Also print the top 3 beam candidates with their scores
  - Example usage: python infer.py --text "Hello, how are you?"
 
================================================================================
PROMPT 6 — BASELINE MODEL (for Academic Comparison)
================================================================================
 
Write a baseline/train_baseline.py script that fine-tunes IndicTrans2
as a standalone model on the same 300k corpus for comparison.
 
Requirements:
  - Load ai4bharat/indictrans2-en-indic-dist-200M as AutoModelForSeq2SeqLM
  - Use the same train/val/test split as the hybrid model
  - Use HuggingFace Seq2SeqTrainer for simplicity
  - src_lang: eng_Latn, tgt_lang: mni_Mtei
  - Training args: batch_size=32, epochs=10, fp16=True, LR=5e-5
  - Evaluate with BLEU and chrF++ on the same test set
  - Save results to results/baseline_results.json
 
This baseline will be used to show whether the hybrid model
(IndicBERT encoder + IndicTrans2 decoder) outperforms or underperforms
vs. IndicTrans2 used as a complete end-to-end system.
 
Also write results/compare.py that loads both results files and prints
a side-by-side comparison table of BLEU, chrF++, and COMET scores
for: (1) Hybrid model, (2) IndicTrans2 baseline
 
================================================================================
ARCHITECTURE QUICK REFERENCE
================================================================================
 
Component    | Model                              | Output Shape           | Trainable
-------------|-------------------------------------|------------------------|-------------------------
Encoder      | IndicBERT v2 MLM-Sam-TLM           | [batch, seq_len, 768]  | Top 4 layers (Phase 3)
Bridge       | Linear(768→1024) + LayerNorm        | [batch, seq_len, 1024] | All phases (from scratch)
Decoder      | IndicTrans2 dist-200M decoder only  | [batch, tgt_len, vocab]| Top 4 layers (Phase 2+3)
 
================================================================================
3-PHASE TRAINING SCHEDULE
================================================================================
 
Phase   | What Trains                          | Epochs  | LR    | Goal
--------|--------------------------------------|---------|-------|---------------------
Phase 1 | Projection bridge only               | 1 - 5   | 5e-4  | Stabilize bridge
Phase 2 | Bridge + top 4 decoder layers        | 6 - 10  | 1e-4  | Adapt decoder
Phase 3 | Bridge + top 4 dec + top 4 enc       | 11 - 20 | 5e-5  | Full fine-tune
 
================================================================================
CRITICAL WARNINGS
================================================================================
 
1. DIMENSION MISMATCH: IndicBERT outputs 768-dim. IndicTrans2 decoder expects
   1024-dim. The ProjectionBridge is mandatory — never skip it.
 
2. TWO TOKENIZERS: IndicBERT tokenizer for English input. IndicTrans2 tokenizer
   for Meitei output. Never mix them.
 
3. LANGUAGE TOKEN: Always force BOS token = mni_Mtei in decoder.
   Without it, output language is unpredictable.
 
4. NO INDICTRANS2 ENCODER: Extract ONLY model.model.decoder and model.lm_head
   from IndicTrans2. Discard its encoder entirely.
 
5. PHASE ORDER MATTERS: Never unfreeze all layers at once. Always start with
   only the bridge, or pretrained weights will be destroyed by random gradients.
 
================================================================================
Run Prompt 1 first → confirm both models download → then proceed in order.
================================================================================
 